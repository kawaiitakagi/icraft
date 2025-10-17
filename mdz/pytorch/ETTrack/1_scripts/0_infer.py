import os
import sys
import time
from collections import OrderedDict
import types

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import _save_tracker_output
from pytracking.evaluation import Tracker

if __name__ == '__main__':
    dataset_name = 'lasot'
    tracker_name = 'et_tracker'
    tracker_param = 'et_tracker'
    visualization=None
    debug=None
    visdom_info=None
    run_id = 2405101500_2
    dataset = get_dataset(dataset_name)

    tracker = Tracker(tracker_name, tracker_param, run_id)

    params = tracker.get_parameters()
    visualization_ = visualization

    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    if visualization is None:
        if debug is None:
            visualization_ = getattr(params, 'visualization', False)
        else:
            visualization_ = True if debug else False

    params.visualization = visualization_
    params.debug = debug_

    for seq in dataset[:]:
        print(seq)
        def _results_exist():
            if seq.dataset == 'oxuva':
                vid_id, obj_id = seq.name.split('_')[:2]
                pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
                return os.path.isfile(pred_file)
            elif seq.object_ids is None:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
                return os.path.isfile(bbox_file)
            else:
                bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
                missing = [not os.path.isfile(f) for f in bbox_files]
                return sum(missing) == 0

        visdom_info = {} if visdom_info is None else visdom_info

        if _results_exist() and not debug:
            print('FPS: {}'.format(-1))
            continue

        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

        tracker._init_visdom(visdom_info, debug_)
        if visualization_ and tracker.visdom is None:
            tracker.init_visualization()

        # Get init information
        init_info = seq.init_info()
        et_tracker = tracker.create_tracker(params)
        output = {'target_bbox': [],
            'time': [],
            'segmentation': [],
            'object_presence_score': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = tracker._read_image(seq.frames[0])

        if et_tracker.params.visualization and tracker.visdom is None:
            tracker.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = et_tracker.initialize(image, init_info)
        
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'object_presence_score': 1.}

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = tracker._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = et_tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if tracker.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif et_tracker.params.visualization:
                tracker.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        output['image_shape'] = image.shape[:2]
        output['object_presence_score_threshold'] = et_tracker.params.get('object_presence_score_threshold', 0.55)

        sys.stdout.flush()

        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        if not debug:
            _save_tracker_output(seq, tracker, output)