import os
import os.path as osp
import argparse


if __name__ == '__main__':
    # get agrs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()

    keys = args.model.split('_')
    assert keys[0] in ('TecoGAN', 'FRVSR')
    assert keys[1] in ('BD', 'BI')

    # set dirs
    Vid4_GT_dir = 'data/Vid4/GT'
    Vid4_SR_dir = 'results/Vid4/{}'.format(args.model)
    Vid4_vids = ['calendar', 'city', 'foliage', 'walk']

    ToS3_GT_dir = 'data/ToS3/GT'
    ToS3_SR_dir = 'results/ToS3/{}'.format(args.model)
    ToS3_vids = ['bridge', 'face', 'room']

    # evaluate Vid4
    if osp.exists(Vid4_SR_dir):
        Vid4_GT_lst = [
            osp.join(Vid4_GT_dir, vid) for vid in Vid4_vids]
        Vid4_SR_lst = [
            osp.join(Vid4_SR_dir, vid) for vid in Vid4_vids]
        os.system('python codes/official_metrics/metrics.py --output {} --results {} --targets {}'.format(
            osp.join(Vid4_SR_dir, 'metric_log'),
            ','.join(Vid4_SR_lst),
            ','.join(Vid4_GT_lst)))

    # evaluate ToS3
    if osp.exists(ToS3_SR_dir):
        ToS3_GT_lst = [
            osp.join(ToS3_GT_dir, vid) for vid in ToS3_vids]
        ToS3_SR_lst = [
            osp.join(ToS3_SR_dir, vid) for vid in ToS3_vids]
        os.system('python codes/official_metrics/metrics.py --output {} --results {} --targets {}'.format(
            osp.join(ToS3_SR_dir, 'metric_log'),
            ','.join(ToS3_SR_lst),
            ','.join(ToS3_GT_lst)))

