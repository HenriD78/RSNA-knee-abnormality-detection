# MAE inference submission

Only this guide and the aggregate verification receipt are versioned. The model assets, codec wheels, ZIPs and example predictions described below must be supplied locally; they are excluded from Git. Notebook: [MAE inference](../notebooks/mae/mae_submission.ipynb).

Deliverables:

- `../notebooks/mae/mae_submission.ipynb`: self-contained inference notebook.
- `mae_assets.zip`: uploadable encoder and five-classifier bundle, about 7 MB compressed.
- `dicom_codecs.zip`: offline GDCM decoder wheels for Linux x86_64 Python 3.10–3.13, about 48 MB. Upload as a separate dataset.
- `local_example/submission.csv`: predictions for the three supplied example test studies, using all 557 existing JPEG slices.
- `verification.json` and `preprocessing_audit.csv`: local verification results.

## Run on Kaggle

1. Create a Kaggle dataset from `mae_assets.zip`. Ensure that its contents are extracted: `encoder.pt`, `config.json`, `manifest.json`, and five `probe_fold*.pt` files.
2. Create another Kaggle dataset from `dicom_codecs.zip`, named **RSNA DICOM Codecs**, and ensure the wheels and `codecs_manifest.json` are extracted. Import the updated `mae_submission.ipynb`. Attach the competition, the model asset dataset, and the codec dataset. Restart the session to clear any cached pydicom decoder availability from an earlier run.
3. Choose a GPU accelerator and disable Internet. If automatic discovery is ambiguous, set `COMPETITION_DIR_OVERRIDE` and `ASSET_DIR_OVERRIDE` in the first code cell to the corresponding mounted directories.
4. Save and run all cells. Inspect `inference_report.json` and `decode_errors.csv`, then submit `/kaggle/working/submission.csv` through the competition notebook interface.

The notebook reads `test.csv` on every run, so it processes the hidden studies when Kaggle reruns it. Do not submit the local example CSV as a substitute for a notebook rerun. No Kaggle upload, run, or competition submission has been performed by this local workflow.

### Submit the completed notebook version through the CLI

After uploading and successfully running the notebook, use its Kaggle URL slug and saved version number. For example, **if** the URL is `https://www.kaggle.com/code/henridargent/rsna-knee-mae-inference` and the completed version is `1`:

```sh
.venv/bin/kaggle competitions submit rsna-knee-abnormality-detection \
  -f submission.csv \
  -k henridargent/rsna-knee-mae-inference \
  -v 1 \
  -m "Frozen MAE, mean pooling, five-fold classifier ensemble"
```

These slug/version values are examples, not an existing uploaded notebook. The installed CLI accepts the competition name as a positional argument; its help does not list `-c`. If another installed CLI version supports `-c`, that form may be used there. For a code competition, `-f submission.csv` refers to the output of the specified Kaggle notebook version; it does not upload our local example CSV. This command does not create the notebook or attach its weights.

PyTorch, NumPy, pandas, Pillow, OpenCV, and pydicom 3.x are required. The first cell installs the matching `python-gdcm==3.0.26` Linux wheel from the attached codec dataset into `/kaggle/working/_dicom_codecs`, before importing pydicom. Installation uses `--no-index --no-deps`; it neither downloads anything nor replaces NumPy. GDCM supports the JPEG Lossless syntax that was missing in the initial Kaggle run and JPEG 2000. The subsequent decoder check remains enabled.

The codec bundle contains Python 3.10, 3.11, 3.12 and 3.13 wheels from PyPI, with SHA-256 hashes. Selection, hashes, wheel metadata and error handling are checked locally. Linux binary execution cannot be tested on this Mac; rerun on Kaggle to confirm that environment. If the notebook still shows the old error text, verify that the updated first cell starts with `# Offline codec setup` and that you restarted the session.

## What the model does

The source is `outputs/mae_baseline/pilot_20260924T083354_091484Z/best.pt`: the saved reconstruction-selected checkpoint from the 30-epoch run. Its encoder processes each image without masking, averages patch embeddings, then averages all image features within each study. Each of the five saved classifiers applies its own training-fold standardization and logistic weights. Final probabilities are the arithmetic mean of the five fold probabilities.

No retraining is performed. The bundle strips the reconstruction decoder, optimizer, reports, images, and study identifiers. The original training notebook and checkpoints are unchanged.

The reported local OOF macro AUC of 0.5741 evaluates held-out predictions on only 58 labeled studies. It is not a public or private leaderboard score, and it is not a held-out evaluation of the five-fold ensemble. Kaggle must run the notebook to provide a public leaderboard score.

## Pixel conversion and remaining limits

The original JPEG conversion source was not available. A comparison with 25 matching local DICOM/JPEG pairs identified this near-exact conversion:

1. Decode raw grayscale pixels to float32.
2. Min–max normalize each frame and convert to uint8.
3. Resize to 224 × 224 with OpenCV `INTER_AREA`.
4. Encode at JPEG quality 95, decode with Pillow, and divide by 255.

Mean pixel error was approximately 0.00138 on the 0–255 scale; the largest mean error for an individual slice was 0.00769. These small nonzero differences may reflect converter/library differences. This empirical check does not establish exact matching for every scanner. No new windowing, physical crop, orientation normalization, or photometric inversion is introduced without evidence that it existed during training.

The actual paired DICOM files use uncompressed Explicit VR Little Endian. Synthetic JPEG 2000 and RLE multi-frame decoding are also checked. These checks do not reproduce Kaggle's full hidden set or its installed library versions. The final runtime on Kaggle remains unmeasured.

Corrupt slices are logged and skipped. A study with files but no decodable frames receives neutral 0.5 probabilities and a warning. Missing files, missing codec dependencies, or encoder failures stop the run. Study pooling uses all successfully decoded slices; no slice-count cap is imposed.

## Recheck locally

From the project root:

```sh
.venv/bin/python scripts/verification/verify_mae_submission.py
```

This executes the notebook in a Jupyter kernel, checks classifier predictions against all saved OOF predictions, compares the trimmed encoder and full-study features with the original model, tests the paired pixels, and exercises DICOM inference and error handling. It writes an executed notebook under `local/scratch/` and keeps partial-DICOM fixture predictions separate from the complete example-JPEG submission.
