import numpy as np

from .detection.iou import DetectionIoUEvaluator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric():
    def __init__(self, is_output_polygon=False):
        # is_output_polygon：是否输出的是个多边形
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon) # 评估器：核心 ★★★

    def measure(self, batch, output, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        box_thresh : 文字区域 和 gt 的 iou 阈值
        output: (polygons, ...)
        '''
        results = []
        # 因为传入的是一个 batch 也是若干图片，所以
        gt_polyons_batch = batch['text_polys'] # ground truth，这里代表有若干图片，每张图片有若干文字框的 gt每个文字框都
        ignore_tags_batch = batch['ignore_tags'] #
        pred_polygons_batch = np.array(output[0]) # 同理，只不过是pred的框
        pred_scores_batch = np.array(output[1])
        # for循环遍历每一张图片，得到这一张图片里的所有多边形框，包括gt和预测的框还有分数等等类容
        for polygons, pred_polygons, pred_scores, ignore_tags in zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            # gt 整理一下，一个列表，每个元素是个字典，代表对应文字框的点坐标和是否忽略
            gt = [dict(points=np.int64(polygons[i]), ignore=ignore_tags[i]) for i in range(len(polygons))]
            if self.is_output_polygon:
                # 是多边形输出直接整理成列表形式
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                # print(pred_polygons.shape)
                # 如果输出的要求是矩形，四边形，那么需要判断一下分数是否大于thresh，然后再整理
                # 为什么输出的是举行的时候要加一步判断，比如有一种特殊情况，个人理解ppt
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i, :, :].astype(np.int)))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics): # 把所有batch整个一下，求得整个数据据上的评估结果
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
