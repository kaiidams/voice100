from .models.align import AudioAlignCTC

def main_cli():
    model = AudioAlignCTC.load_from_checkpoint('model/align_en_lstm_base_ctc.ckpt')
    model