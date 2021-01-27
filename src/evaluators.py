'''
Module that contains evaluation classes
'''

import numpy as np

import utils


class Evaluator:
    '''
    Abstract object detection evaluator
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Initialize the accumulated results
        '''
        self.results = []

    def update(self, result):
        '''
        Update the current results with a list containing
        one (output, target) pair for each evaluated image
        '''
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        self.results.extend(result)

    @property
    def prefix(self):
        '''
        This function should return a string representing
        a possible prefix to prepend to computed metrics
        (e.g. for logging purposes)
        '''
        raise NotImplementedError()

    @property
    def default_scores(self):
        '''
        This function should return a dictionary
        with as key the name of a metric and as
        value its default one
        '''
        raise NotImplementedError()

    def evaluate(self):
        '''
        This function should return a dictionary
        with key-value pairs the metric name
        and its computed value
        '''
        raise NotImplementedError()


class PascalEvaluator(Evaluator):
    '''
    Object detection evaluator respecting
    the Pascal VOC competition metrics
    '''

    IOU_THRESH = 0.5

    def __init__(self):
        super().__init__()

    @property
    def prefix(self):
        return 'pascal'

    @property
    def default_scores(self):
        return {'tps': 0, 'fps': 0, 'gts': 0, 'ap': 0.0}

    def average_precision(self, precisions, recalls):
        '''
        Compute the average precision score as the sum of
        precisions, weighted by the difference of the current
        and previous recalls

        See https://github.com/rafaelpadilla/Object-Detection-Metrics
        '''
        # Append sentinel values to beginning and end
        prec = [0] + precisions + [0]
        rec = [0] + recalls + [1]

        # Maximum precision values
        for i in range(len(prec) - 1, 0, -1):
            prec[i - 1] = max(prec[i - 1], prec[i])

        # Get recall indexes
        indexes = []
        for i in range(len(rec) - 1):
            if rec[1:][i] != rec[0:-1][i]:
                indexes.append(i + 1)

        # Compute all-points average precision
        ap = 0
        for i in indexes:
            ap = ap + np.sum((rec[i] - rec[i - 1]) * prec[i])

        return ap

    def evaluate(self):
        '''
        Mainly computes average precision at the specified
        IoU threshold, as per Pascal VOC evaluation
        '''
        num_gts, num_dets = 0, 0
        whole_confs, whole_dets, whole_tps = [], [], []

        # For each processed image
        for i, (output, target) in enumerate(self.results):

            # Initialize ground-truth related stuff
            gts = len(target["boxes"])
            already_detected = np.full(gts, False)
            num_gts += gts

            # Initialize detection related stuff
            dets = (output["labels"] != 0)
            confs = np.full(sum(dets), 0.0)
            tps = np.full(sum(dets), False)
            num_dets += sum(dets)

            # For each output, find the ground truth with which
            # it overlaps the most and keep it if IoU is greater
            # than a fixed threshold
            for j, (box, score) in enumerate(zip(output["boxes"], output["scores"])):
                # If no background is detected (found a table)
                if dets[j]:
                    best_match = utils.most_overlapping_box(
                        box, target["boxes"], self.IOU_THRESH
                    )
                    if best_match is not None:
                        target_index, _, _ = best_match
                        if not already_detected[target_index]:
                            already_detected[target_index] = True
                            confs[i], tps[i] = score, True

            # Store info
            whole_confs.extend(confs)
            whole_dets.extend(dets)
            whole_tps.extend(tps)

        # Convert lists to numpy
        confs, dets, tps = (
            np.array(whole_confs),
            np.array(whole_dets),
            np.array(whole_tps)
        )

        # Sort values by confidence scores
        sorted_indices = np.argsort(-confs)
        confs = confs[sorted_indices]
        tps = tps[sorted_indices]
        dets = dets[sorted_indices]

        # Compute false positives and
        # cumulative true and false positives
        fps = np.invert(tps)
        cum_tps = np.cumsum(tps)
        cum_fps = np.cumsum(fps)

        # Compute average precision score
        precisions = cum_tps / (cum_tps + cum_fps)
        recalls = cum_tps / (num_gts + 1e-16)
        ap = self.average_precision(precisions, recalls)

        return {
            'tps':  np.sum(tps),
            'fps': np.sum(fps),
            'gts': num_gts,
            'ap': ap,
        }


class ICDAR19Evaluator(Evaluator):
    '''
    Object detection evaluator respecting
    the ICDAR19 competition metrics
    '''

    IOU_RANGE = np.arange(0.6, 1.0, 0.1)

    def __init__(self):
        super().__init__()

    @property
    def prefix(self):
        return 'icdar19'

    @property
    def default_scores(self):
        return {'wp': 0.0, 'wr': 0.0, 'wf1': 0.0}

    def metrics_at_iou(self, predictions, ground_truths, iou):
        '''
        Given and IoU threshold T, return the number 
        of TP@T, FP@T and FN@T as following:
        - The TP@T is the number of ground truth tables that have 
            a major overlap (IoU >= T) with one of the detected tables 
        - The FP@T indicates the number of detected tables that 
            do not overlap (IoU < T) with any of the ground tables
        - The FN@T indicates the number of ground truth tables that 
            do not overlap (IoU < T) with any of the detected tables
        '''
        tps = np.full(len(ground_truths), False)
        fps = np.full(len(predictions), True)

        for box_index, box in enumerate(predictions):
            best_match = utils.most_overlapping_box(
                box, ground_truths, iou
            )
            if best_match is not None:
                gt_index, _, _ = best_match
                fps[box_index] = False
                tps[gt_index] = True

        fns = np.invert(tps)
        return np.sum(tps), np.sum(fps), np.sum(fns)

    def evaluate(self):
        '''
        Evaluate predictions as done in the ICDAR19 competition
        (see http://sac.founderit.com/evaluation.html)
        '''
        wp, wr, wf1 = 0.0, 0.0, 0.0
        for iou in self.IOU_RANGE:

            # For each processed image, aggregate TP, FP, FN
            tps_at_iou, fps_at_iou, fns_at_iou = 0, 0, 0
            for predictions, ground_truths in self.results:
                tps, fps, fns = self.metrics_at_iou(
                    predictions["boxes"], ground_truths["boxes"], iou
                )
                tps_at_iou += tps
                fps_at_iou += fps
                fns_at_iou += fns

            # Compute precision, recall and F1 at the current
            # IoU threshold
            p_at_iou = tps_at_iou / (fps_at_iou + fns_at_iou)
            r_at_iou = tps_at_iou / (fns_at_iou + tps_at_iou)
            f1_at_iou = (
                2 * tps_at_iou /
                (fps_at_iou + fns_at_iou + 2 * tps_at_iou)
            )

            # Aggregate metrics into weighted averages
            wp += p_at_iou * iou
            wr += r_at_iou * iou
            wf1 += f1_at_iou * iou

        # Divide weighted metrics by the sum of weights
        weights = np.sum(self.IOU_RANGE)
        wp /= weights
        wr /= weights
        wf1 /= weights

        return {'wp': wp, 'wr': wr, 'wf1': wf1}


class AggregatedEvaluator:
    '''
    Utility class that aggregates metrics coming
    from different evaluation strategies
    '''

    def __init__(self, evaluators):
        self.reset()
        self.evaluators = []
        for evaluator in evaluators:
            if isinstance(evaluator, type):
                self.evaluators.append(evaluator())
            elif isinstance(evaluator, Evaluator):
                self.evaluators.append(evaluator)

    def reset(self):
        '''
        Initialize the accumulated results
        '''
        self.results = []

    def update(self, result):
        '''
        Update the current results with a list containing
        one (output, target) pair for each evaluated image
        '''
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        self.results.extend(result)

    def evaluate(self):
        '''
        Aggregate all the metrics computed by different evaluators
        and return a single dictionary
        '''
        metrics = dict()
        for evaluator in self.evaluators:
            evaluator.results = self.results
            evaluator_metrics = {
                f'{evaluator.prefix}/{k}': v
                for k, v in evaluator.evaluate().items()
            }
            metrics.update(evaluator_metrics)
        return metrics
