"""Noms publics conservés pour rester compatible avec les CSV RSNA."""

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
PLANES = ["Sagittal", "Coronal", "Axial"]
READERS = {
    "coatnet": {"model": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "size": 384},
    "small": {"model": "convnext_small.fb_in22k_ft_in1k", "size": 224},
    "tiny": {"model": "convnext_tiny.fb_in22k_ft_in1k", "size": 224},
    "dino": {"model": "vit_small_patch14_dinov2.lvd142m", "size": 224},
}
TINY_WEIGHTS = {
    "ACL": .25, "MCL": .20, "Medial Meniscus": .20, "Lateral Meniscus": .08,
    "Medial OA": .25, "Lateral OA": .08, "PF OA": .08, "Effusion": .04,
    "Synovitis": .03, "Baker's": .35, "Contusion": .30, "Fracture": .08,
}
