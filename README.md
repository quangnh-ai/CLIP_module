# CLIP

## Extract Keyframes

```bash
python extract_keyframes.py --keyframe_folder_path 'path to keyframes folder' \
--save_path 'path to save keyframe' \
--cuda
```

keyframe folder structure:
```bash
keyframe
 |
 |-0000
 |  |-frames01.jpg
 |  |-frames02.jpg
 |  |-...
 |-0001
 |-...
```
For more example about the keyframes folder structure, you can open the thumbs folder inside data folder.

## Mini demo
```bash
python demo.py
```

## API
```bash
python app.py --features_path keyframes.h5
```