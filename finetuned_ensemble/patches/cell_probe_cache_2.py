# Run after the updated recovery cell: measure only TWO centres.
from rsna3.common import read_json
centres = 2
probe_path = WORK / 'probe_cache_2_lossless.json'
cli('probe-cache', '--studies-csv', PROJECT / 'data/train.csv',
    '--series-csv', COMPETITION_ROOT / 'train_series.csv',
    '--dicom-root', dicom_directory('train'), '--output', probe_path,
    '--centres', centres, '--n', 75, '--max-hours', MAX_CPU_HOURS)
profile = read_json(probe_path)
print(json.dumps({
    'probe_file': str(probe_path),
    'estimated_hours_with_margin': round(profile['projected_seconds_with_margin'] / 3600, 2),
    'estimated_gib_with_margin': round(profile['projected_bytes_with_margin'] / (1 << 30), 2),
    'fits_8h_cpu_limit': profile['projected_seconds_with_margin'] <= MAX_CPU_HOURS * 3600,
    'fits_12gib_storage_limit': profile['projected_bytes_with_margin'] <= 12 * (1 << 30),
}, indent=2))
