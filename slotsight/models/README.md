# Detection models

This folder holds downloaded detection models. Git ignores everything here except this file.

- On first run, `python slotsight/app.py` downloads the default model, **YOLO11 small** (`yolo11s.pt`, 19 MB), from Ultralytics' GitHub release. Until it finishes, the dashboard shows "Downloading" and every slot shows as unknown.
- To install other models, use **Configure slots > Detection model** in the dashboard, or a terminal:

  ```powershell
  python slotsight/scripts/models.py list
  python slotsight/scripts/models.py install huggingface/rtdetr-v2-r18
  python slotsight/scripts/models.py use huggingface/rtdetr-v2-r18
  ```

- Every download is checked against a published SHA-256 or MD5 hash. Once installed, a model works offline.
- To use your own Ultralytics weights, put the `.pt` file here and set `"model": "ultralytics/file:my_model.pt"` in `config/settings.txt`.

## Available models and their licences

All of them are trained on COCO. SlotSight uses only their car, motorcycle, bus and truck detections.

| Id | Provider | Weights licence | Download | Python package needed |
| --- | --- | --- | --- | --- |
| `ultralytics/yolo11n`, `yolo11s` (default), `yolo11m`, `yolo26n`, `yolo26s` | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | 6 to 41 MB | `ultralytics` |
| `huggingface/rtdetr-v2-r18` | [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd) on Hugging Face | Apache-2.0 | 81 MB | `transformers` |
| `rfdetr/nano`, `rfdetr/small` | [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) | Apache-2.0 | 366 to 386 MB | `rfdetr` |
| `opencv/yolox-s` | [OpenCV model zoo](https://huggingface.co/opencv/object_detection_yolox) (YOLOX-S) | Apache-2.0 | 36 MB | none (uses OpenCV) |

## The Ultralytics licence (AGPL-3.0)

The `ultralytics` package and its YOLO weights are licensed under the [GNU AGPL-3.0](https://www.ultralytics.com/license). AGPL-3.0 is a strong copyleft licence. If you distribute software that uses them, or let other people use it over a network (for example, a hosted dashboard), you must make the complete source code of that software available under AGPL-3.0. Commercial use without that obligation needs an [Ultralytics Enterprise licence](https://www.ultralytics.com/license).

SlotSight is a local, single-user prototype, so running it on your own laptop for your own use doesn't trigger these obligations. Sharing or hosting it does. To avoid AGPL code entirely, install one of the Apache-2.0 models and uninstall `ultralytics`.

This is a summary, not legal advice. The licence texts are what count.
