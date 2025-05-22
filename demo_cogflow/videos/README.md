# Video Production Workflow

This directory contains all the parts and instructions needed to assemble the final demo video (`output.mp4`).

## Workflow Steps

### 1. Update Intro/Outro Slides (if needed)
- Place your new intro or outro video in this directory (e.g., `0_verge_intro.mp4`, `7_final_slide.mp4`).

### 2. Convert to MKV with AAC Audio
For each new or updated part (e.g., intro/outro), convert to MKV with AAC audio:

```bash
ffmpeg -i 0_verge_intro.mp4 -c:v copy -c:a aac 0_verge_intro_aac.mkv
ffmpeg -i 7_final_slide.mp4 -c:v copy -c:a aac 7_final_slide_aac.mkv
```

### 3. Update `filelist.txt`
Edit `filelist.txt` to list all the video parts in the correct order, using the `_aac.mkv` versions. Example:

```
file '0_verge_intro_aac.mkv'
file '1_introduction_and_overview_aac.mkv'
file '2_login_explain_overview_aac.mkv'
file '3_open_cf_server_code_and_run_aac.mkv'
file '4_pipeline_in_process_aac.mkv'
file '5_pipeline_serving_success_1280_720_aac.mkv'
file '6_outro_aac.mkv'
file '7_final_slide_aac.mkv'
```

### 4. Concatenate All Parts into Final Output
Run the following command to produce the final video:

```bash
ffmpeg -f concat -safe 0 -i filelist.txt -c:v libx264 -c:a aac output.mp4
```

## Notes
- Only the `_aac.mkv` files are used for the final output.
- The original `.mp4` or `.mkv` files can be kept for reference or deleted to save space.
- You can preview the final video with any standard video player.

---

**Contact:** For questions, contact the project maintainer. 