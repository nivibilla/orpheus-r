
import torch
from snac import SNAC

class OrpheusAudioDecoder:
    def __init__(self, device='cpu'):
        self.snac_model = SNAC.from_pretrained("srinivasbilla/snac_24khz").eval()
        self.device = device
        self.snac_model.to(self.device)

    def decode(self, output_text):
        token_gen = ['<' + token for token in output_text.split('<') if token]
        buffer = []
        count = 0
        for token_sim in token_gen:
            token = self.turn_token_into_id(token_sim, count)
            if token is None:
                pass
            else:
                if token > 0:
                    buffer.append(token)
                    count += 1
        decoded_audio = self.convert_to_audio(buffer, self.device)
        if decoded_audio is not None:
            return decoded_audio
        else:
            print("Decoding failed.")
            return

    def convert_to_audio(self, multiframe, snac_device):
        frames = []
        if len(multiframe) < 7:
            return

        codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        for j in range(num_frames):
            i = 7 * j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor(
                    [frame[i]], device=snac_device, dtype=torch.int32
                )
            else:
                codes_0 = torch.cat(
                    [
                        codes_0,
                        torch.tensor([frame[i]], device=snac_device, dtype=torch.int32),
                    ]
                )

            if codes_1.shape[0] == 0:

                codes_1 = torch.tensor(
                    [frame[i + 1]], device=snac_device, dtype=torch.int32
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 1]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )

            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor(
                    [frame[i + 2]], device=snac_device, dtype=torch.int32
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 2]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=snac_device, dtype=torch.int32
                        ),
                    ]
                )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # check that all tokens are between 0 and 4096 otherwise return *
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > 4096)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > 4096)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > 4096)
        ):
            del codes_0
            del codes_1
            del codes_2
            del codes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)

            detached_audio = audio_hat.squeeze().detach().cpu().numpy()

            audio_np = detached_audio

            # audio_int16 = (audio_np * 32767).astype(np.int16)
            # audio_bytes = audio_int16.tobytes()

            del audio_hat
            del detached_audio
            del codes_0
            del codes_1
            del codes_2
            del codes

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return audio_np

    @staticmethod
    def turn_token_into_id(token_string, index):
        # Strip whitespace
        token_string = token_string.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")

        if last_token_start == -1:
            print("No token found in the string")
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        else:
            return None
