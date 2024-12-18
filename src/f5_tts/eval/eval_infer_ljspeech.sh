# !/bin/bash

model_name="F5TTS_Small_beta_vocos_char_LJSpeech_train"
file_list="src/f5_tts/eval/ljs_audio_text_test_filelist.txt"

ref_audio=/store/store4/data/LJSpeech-1.1/wavs/LJ022-0023.wav
ref_text="The overwhelming majority of people in this country know how to sift the wheat from the chaff in what they hear and what they read."

# Read the file line by line
while IFS='|' read -r filepath text; do
  # Extract the filename (without the .wav extension) and the text
  filename=$(basename "$filepath" .wav)

  # Print the extracted information (you can modify this part to do whatever you need)
  echo "Filename: $filename"
  echo "Text: $text"
  echo "--------------------"
  f5-tts_infer-cli --model F5-TTS --ref_audio $ref_audio --ref_text "$ref_text" --gen_text "$text" --output_dir tests/$model_name --output_file ${filename}.wav --ckpt_file "ckpts/$model_name/model_last.pt" --model_cfg "ckpts/$model_name/config.yaml" --vocab_file "data/LJSpeech_train_char/vocab.txt"

  # Example: Create a directory for each filename and save the text in a file
  # directory="output/$filename"
  # mkdir -p "$directory"
  # echo "$text" > "$directory/$filename.txt"

done < "$file_list"

echo "Infer complete."
