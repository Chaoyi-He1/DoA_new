import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .positional_embedding import build_position_encoding


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
        The process happens in two steps:
            1) we compute hungarian assignment between ground truth and the outputs of the model
            2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.register_buffer('weight_dict', weight_dict)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'quadrant': self.loss_quadrant,
            'directions': self.loss_directions,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2).contiguous(), 
                                  target_classes, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_directions(self, outputs, targets, indices, num_directions):
        """Compute the losses related to the angle: the L1 regression loss.
           targets dicts must contain the key "directions" containing a tensor of dim [nb_target_boxes, angle]
        """
        assert "pred_directions" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_directions = outputs['pred_directions'][idx]
        target_directions = torch.cat([t['directions'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_directions = F.cross_entropy(src_directions, 
                                          target_directions, reduction='none')

        losses = {'loss_directions': loss_directions.sum() / num_directions}
        losses['direction_error'] = 100 - accuracy(src_directions, target_directions)[0]
        return losses
    
    def loss_quadrant(self, outputs, targets, indices, num_quadrants):
        """Compute the losses related to the quadrant: the BCE loss.
           targets dicts must contain the key "quadrant" containing a tensor of dim [nb_target_boxes, quadrant]
        """
        assert "pred_directions" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_quadrant = outputs['pred_quadrant'][idx]
        target_quadrant = torch.cat([t['quadrant'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_quadrant = F.binary_cross_entropy_with_logits(src_quadrant, target_quadrant, reduction='none')

        losses = {'loss_quadrant': loss_quadrant.sum() / num_quadrants}
        losses['quadrant_error'] = 100 - accuracy(src_quadrant, target_quadrant)[0]
        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __int__(self):
        super(PostProcess, self).__int__()
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    
    
class Detection(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim * 2, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim * 2, 4, 3)
        self.quadrant_embed = MLP(hidden_dim, hidden_dim * 2, 1, 3)
        self.direction_embed = MLP(hidden_dim, hidden_dim * 2, 91, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embed = build_position_encoding('sine', hidden_dim)
    
    def forward(self, inputs):
        features = self.backbone(inputs)
        pos = self.pos_embed(features)
        hs = self.transformer(tgt=self.query_embed.weight.unsqueeze(0).repeat(inputs.shape[0], 1, 1), 
                              memory=features, 
                              pos=pos)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        # max_coord = outputs_coord.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        outputs_coord = outputs_coord.sigmoid()
        outputs_quadrant = self.quadrant_embed(hs).sigmoid()
        outputs_direction = self.direction_embed(hs)
        # max_direction = outputs_direction.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        # outputs_direction = (outputs_direction / max_direction).sigmoid()
        
        out = {'pred_logits': outputs_class, 
               'pred_boxes': outputs_coord, 
               'pred_quadrant': outputs_quadrant, 
               'pred_directions': outputs_direction}
        return out


def build(hyp):
    backbone = build_backbone(hyp)
    transformer = build_transformer(hyp)
    detection = Detection(backbone, 
                          transformer, 
                          hyp['num_classes'], 
                          hyp['num_queries'])
    matcher = build_matcher(hyp)
    
    weight_dict = {'loss_ce': hyp['ce_loss_coef'], 
                   'loss_bbox': hyp['bbox_loss_coef'], 
                   'loss_giou': hyp['giou_loss_coef'], 
                   'loss_quadrant': hyp['quadrant_loss_coef'], 
                   'loss_directions': hyp['direction_loss_coef'],
                   }
    losses = ['labels', 'boxes', 'quadrant', 'directions']   # 
    
    criterion = SetCriterion(hyp['num_classes'],
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=hyp['eos_coef'],
                             losses=losses)
    coco_postprocessors = {'bbox': PostProcess()}
    return detection, criterion, coco_postprocessors
