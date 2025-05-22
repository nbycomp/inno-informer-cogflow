import time
import logging
import assemblyai as aai
import os
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AssemblyAITranscriber:
    def __init__(self):
        self.project_id = "savvy-primacy-390419"  # Your Google Cloud project ID
        self.secret_id = "ASSEMBLY_AI"  # Your secret ID for AssemblyAI API key
        self.api_key ="434aca4d1b1047f48b6d2b4bbbe5729e"
        aai.settings.api_key = self.api_key
        self.transcriber = aai.Transcriber()
        logging.info("AssemblyAI - Initialized with project ID: %s", self.project_id)


    def transcribe_to_text(self, file_url, max_retries=3):
        for attempt in range(max_retries):
            try:
                logging.info(f"AssemblyAI - Starting transcription attempt {attempt + 1} for {file_url}")
                transcript = self.transcriber.transcribe(file_url)
                while transcript.status != 'completed':
                    logging.info("AssemblyAI - Waiting for transcription to complete...")
                    time.sleep(5)
                    transcript = self.transcriber.get(transcript.id)
                logging.info("AssemblyAI - Transcription completed successfully")
                
                # Generate logical filename
                file_name = os.path.basename(file_url).split('.')[0]
                save_path = f"{file_name}_transcription_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
                
                # Save the transcription text to a file
                with open(save_path, 'w') as file:
                    file.write(transcript.text)
                
                logging.info(f"AssemblyAI - Transcription saved to {save_path}")
                return transcript.text
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"AssemblyAI - Transcription attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(5)
                else:
                    logging.error(f"AssemblyAI - Transcription failed after {max_retries} attempts: {e}")
                    raise

    def transcribe_with_word_timing(self, file_url, max_retries=3):
        for attempt in range(max_retries):
            try:
                logging.info(f"AssemblyAI - Starting transcription with word timing attempt {attempt + 1} for {file_url}")
                transcript = self.transcriber.transcribe(file_url)
                while transcript.status != 'completed':
                    logging.info("AssemblyAI - Waiting for transcription to complete...")
                    time.sleep(5)
                    transcript = self.transcriber.get(transcript.id)
                logging.info("AssemblyAI - Transcription with word timing completed successfully")
                return [(word.text, word.start, word.end, word.confidence) for word in transcript.words]
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"AssemblyAI - Transcription with word timing attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(5)
                else:
                    logging.error(f"AssemblyAI - Transcription with word timing failed after {max_retries} attempts: {e}")
                    raise

    def transcribe_and_save_srt_per_word(self, file_url, srt_path):
        logging.info(f"AssemblyAI - Starting transcription and saving SRT per word for {file_url}")
        word_timings = self.transcribe_with_word_timing(file_url)
        srt_content = []
        for index, (word, start, end, _) in enumerate(word_timings, 1):
            start_time = self._ms_to_srt_time(start)
            end_time = self._ms_to_srt_time(end)
            srt_content.append(f"{index}\n{start_time} --> {end_time}\n{word}\n\n")

        with open(srt_path, 'w') as file:
            file.writelines(srt_content)

        logging.info(f"AssemblyAI - SRT file created at {srt_path}")
        return srt_path

    def transcribe_and_save_srt(self, file_url, srt_path, max_retries=3):
        for attempt in range(max_retries):
            try:
                logging.info(f"AssemblyAI - Starting transcription and saving SRT attempt {attempt + 1} for {file_url}")
                transcript = self.transcriber.transcribe(file_url)
                while transcript.status != 'completed':
                    logging.info("AssemblyAI - Waiting for transcription to complete...")
                    time.sleep(5)
                    transcript = self.transcriber.get(transcript.id)
                subtitles_srt = transcript.export_subtitles_srt()

                with open(srt_path, 'w') as srt_file:
                    srt_file.write(subtitles_srt)

                logging.info(f"AssemblyAI - Subtitles file created at {srt_path}")
                return srt_path
            except aai.types.TranscriptError as e:
                if 'transcript id not found' in str(e):
                    logging.warning(f"AssemblyAI - Transcript lookup error: {e}. This might be a temporary issue. Retrying...")
                    time.sleep(5)
                elif attempt < max_retries - 1:
                    logging.warning(f"AssemblyAI - Transcription attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(5)
                else:
                    logging.error(f"AssemblyAI - Transcription failed after {max_retries} attempts: {e}")
                    raise

    @staticmethod
    def _ms_to_srt_time(ms):
        hours = int(ms / 3600000)
        minutes = int((ms % 3600000) / 60000)
        seconds = int((ms % 60000) / 1000)
        milliseconds = int(ms % 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def burn_subtitles(self, video_path, srt_path, output_path):
        """
        Burn subtitles from an SRT file into a video using ffmpeg.
        """
        logging.info(f"Burning subtitles from {srt_path} into {video_path}, outputting to {output_path}")
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}",
            "-c:a", "copy",
            output_path
        ]
        try:
            subprocess.run(command, check=True)
            logging.info(f"Subtitled video created at {output_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to burn subtitles: {e}")

if __name__ == "__main__":
    transcriber = AssemblyAITranscriber()
    file_url = "videos/output.mp4"  # or an audio file
    srt_path = "videos/subtitles.srt"
    transcriber.transcribe_and_save_srt(file_url, srt_path)