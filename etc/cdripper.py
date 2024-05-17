import subprocess
import os
import shutil

def rip_cds():
    output_folder = "you/output/path"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Rip audio from CD using cdparanoia
    try:
        subprocess.run(["cdparanoia", "-B"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error ripping CD:", e)
        return

    # Find the WAV files and move them to the output folder
    wav_files = [f for f in os.listdir() if f.endswith('.wav')]
    for wav_file in wav_files:
        shutil.move(wav_file, os.path.join(output_folder, wav_file))
    print("CD ripped successfully!")

if __name__ == "__main__":
    rip_cds()

