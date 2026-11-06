# Available Metadata Reference

## 1. Tabular Metadata (CSV)

### `train_series.csv` & `test_series.csv`

| Column | Type | Allowed / Sample Values |
| :--- | :--- | :--- |
| `StudyInstanceUID` | String (UID) | `<local-uid>` |
| `SeriesInstanceUID` | String (UID) | `<local-uid>` |
| `Anatomical_Plane` | Categorical | `Sagittal`, `Coronal`, `Axial` |
| `Fluid_Sensitive` | Binary | `0`, `1` |
| `Fat_Suppression` | Binary | `0`, `1` |

---

### `train.csv`

| Column | Type | Values |
| :--- | :--- | :--- |
| `StudyInstanceUID` | String (UID) | `<local-uid>` |
| `Report` | String (Text) | Free-text clinical narrative (Multilingual) |
| `ACL` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `MCL` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Medial Meniscus` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Lateral Meniscus` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Medial OA` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Lateral OA` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `PF OA` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Effusion` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Synovitis` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Baker's` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Contusion` | Binary / Float | `0.0`, `1.0` (or `NaN`) |
| `Fracture` | Binary / Float | `0.0`, `1.0` (or `NaN`) |

---

### `test.csv`

| Column | Type | Sample Value |
| :--- | :--- | :--- |
| `StudyInstanceUID` | String (UID) | `<local-uid>` |

---

## 2. DICOM Header Metadata

### Study & Series Identification

| DICOM Tag | Tag Name | VR | Example Value |
| :--- | :--- | :--- | :--- |
| `(0008,0016)` | `SOPClassUID` | UI | `1.2.840.10008.5.1.4.1.1.4` (MR Image Storage) |
| `(0008,0018)` | `SOPInstanceUID` | UI | `<local-uid>` |
| `(0020,000D)` | `StudyInstanceUID` | UI | `<local-uid>` |
| `(0020,000E)` | `SeriesInstanceUID` | UI | `<local-uid>` |
| `(0020,0052)` | `FrameOfReferenceUID` | UI | `<local-uid>` |
| `(0020,0011)` | `SeriesNumber` | IS | `7` |
| `(0020,0012)` | `AcquisitionNumber` | IS | `1` |
| `(0020,0013)` | `InstanceNumber` | IS | `12` |
| `(0008,103E)` | `SeriesDescription` | LO | `t2_tse_sag_d`, `pd_cor_fs`, `t1_cor` |
| `(0008,0060)` | `Modality` | CS | `MR` |
| `(0008,0008)` | `ImageType` | CS | `['ORIGINAL', 'PRIMARY', 'M', 'NORM']` |

---

### Sequence Physics & Timing Parameters

| DICOM Tag | Tag Name | VR | Example Value |
| :--- | :--- | :--- | :--- |
| `(0018,0020)` | `ScanningSequence` | CS | `SE` (Spin Echo), `GR` (Gradient Recalled) |
| `(0018,0021)` | `SequenceVariant` | CS | `['SP', 'SK']` |
| `(0018,0022)` | `ScanOptions` | CS | `PER`, `FS` |
| `(0018,0023)` | `MRAcquisitionType` | CS | `2D`, `3D` |
| `(0018,0024)` | `SequenceName` | SH | `*tseR2d1_13` |
| `(0018,0080)` | `RepetitionTime` | DS | `5000` (ms) |
| `(0018,0081)` | `EchoTime` | DS | `91` (ms) |
| `(0018,0082)` | `InversionTime` | DS | `None` / `150` (ms) |
| `(0018,0083)` | `NumberOfAverages` | DS | `1` |
| `(0018,0084)` | `ImagingFrequency` | DS | `63.685261` (MHz) |
| `(0018,0086)` | `EchoNumber` | IS | `1` |
| `(0018,0087)` | `MagneticFieldStrength` | DS | `1.5`, `3.0` (Tesla) |
| `(0018,0089)` | `NumberOfPhaseEncodingSteps` | IS | `300` |
| `(0018,0091)` | `EchoTrainLength` | IS | `13` |
| `(0018,0093)` | `PercentSampling` | DS | `75` |
| `(0018,0094)` | `PercentPhaseFieldOfView` | DS | `100` |
| `(0018,0095)` | `PixelBandwidth` | DS | `181` |
| `(0018,1310)` | `AcquisitionMatrix` | US | `[400, 0, 0, 300]` |
| `(0018,1312)` | `InPlanePhaseEncodingDirection` | CS | `COL`, `ROW` |
| `(0018,1314)` | `FlipAngle` | DS | `90`, `180` (degrees) |
| `(0018,1315)` | `VariableFlipAngleFlag` | CS | `N`, `Y` |
| `(0018,0010)` | `ContrastBolusAgent` | LO | `""` |

---

### Spatial Dimensions & Geometry

| DICOM Tag | Tag Name | VR | Example Value |
| :--- | :--- | :--- | :--- |
| `(0028,0010)` | `Rows` | US | `800`, `512` |
| `(0028,0011)` | `Columns` | US | `800`, `512` |
| `(0028,0030)` | `PixelSpacing` | DS | `[0.1875, 0.1875]` ($[dy, dx]$ in mm) |
| `(0018,0050)` | `SliceThickness` | DS | `3.0`, `0.6` (mm) |
| `(0018,0088)` | `SpacingBetweenSlices` | DS | `3.3`, `0.6` (mm) |
| `(0020,0032)` | `ImagePositionPatient` | DS | `[50.001, -131.878, 76.4889]` ($[x, y, z]$ in mm) |
| `(0020,0037)` | `ImageOrientationPatient` | DS | `[r_x, r_y, r_z, c_x, c_y, c_z]` (Direction cosines) |
| `(0020,1041)` | `SliceLocation` | DS | `70.7372` (mm) |

---

### Intensity Representation & VOI LUT

| DICOM Tag | Tag Name | VR | Example Value |
| :--- | :--- | :--- | :--- |
| `(0028,0002)` | `SamplesPerPixel` | US | `1` |
| `(0028,0004)` | `PhotometricInterpretation` | CS | `MONOCHROME2` |
| `(0028,0100)` | `BitsAllocated` | US | `16` |
| `(0028,0101)` | `BitsStored` | US | `12`, `16` |
| `(0028,0102)` | `HighBit` | US | `11`, `15` |
| `(0028,0103)` | `PixelRepresentation` | US | `0` (Unsigned) |
| `(0028,1050)` | `WindowCenter` | DS | `863` |
| `(0028,1051)` | `WindowWidth` | DS | `1711` |
| `(0028,1052)` | `RescaleIntercept` | DS | `0` |
| `(0028,1053)` | `RescaleSlope` | DS | `1` |

---

### Patient & Hardware Context

| DICOM Tag | Tag Name | VR | Example Value |
| :--- | :--- | :--- | :--- |
| `(0010,0020)` | `PatientID` | LO | Pseudo-anonymized string |
| `(0010,0040)` | `PatientSex` | CS | `M`, `F` |
| `(0018,0015)` | `BodyPartExamined` | CS | `KNEE` |
| `(0020,0060)` | `Laterality` | CS | `L`, `R` |
| `(0018,5100)` | `PatientPosition` | CS | `FFS` (Feet First Supine) |
| `(0008,0070)` | `Manufacturer` | LO | `Siemens Healthineers`, `GE`, `Philips` |
| `(0008,1090)` | `ManufacturerModelName` | LO | `MAGNETOM Avanto fit`, `SIGNA HDxt` |
| `(0018,1020)` | `SoftwareVersions` | LO | `syngo MR XA61` |
| `(0018,1250)` | `ReceiveCoilName` | SH | `TxRx_Knee_15` |
| `(0018,1251)` | `TransmitCoilName` | SH | `TxRx_Knee_15` |
