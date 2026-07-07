Set-Location $PSScriptRoot\..

python .\windows_camera_realtime.py `
  --camera-index 0 `
  --face-weights .\yolov5-face\weights\yolov5s-face.pt `
  --emotion-backend emotieff `
  --emotieff-model-name enet_b0_8_best_vgaf `
  --emotieff-engine torch `
  --crop-margin 0.30 `
  --sample-step 3
