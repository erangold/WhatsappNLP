from os import path
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime as dt

whatsapp_datetime_format = "%m/%d/%y, %H:%M"
csv_datetime_format = "%yy-%m-%d, %H:%M:%S"
whatsapp_datetime_reg = "\d{1,2}\/\d{1,2}\/\d{2}, \d{1,2}:\d{2}"
whatsapp_message_format = f"^({whatsapp_datetime_reg}) - ([\w ]+):(.+)"


def read_raw_transcript(raw_file_path, encoding = 'utf8'):
    curr_datetime, curr_sender, curr_message = None, None, ''
    chat_df = pd.DataFrame()
    with open(raw_file_path, 'r', encoding = encoding) as fin:
        with tqdm(fin) as t:
            for line in t:
                line = line.strip()
                matches = re.findall(whatsapp_message_format, line)
                if len(matches) == 0:
                    if curr_message == '<Media omitted>' or curr_message == '':
                        curr_message = line
                    else:
                        curr_message = f"{curr_message}, {line}"
                    continue
                else:
                    if curr_message != '<Media omitted>' and curr_message != '':
                        chat_df = chat_df.append({'timestamp': curr_datetime,
                                                  'sender': curr_sender,
                                                  'message': curr_message}, ignore_index = True)
                    curr_datetime_str, curr_sender, curr_message = matches[0]
                    curr_datetime = dt.strptime(curr_datetime_str, whatsapp_datetime_format)
                    curr_message = curr_message.strip()
                    if curr_message == '<Media omitted>':
                        curr_message == ''
            if curr_message != '':
                chat_df = chat_df.append({'timestamp': curr_datetime,
                                          'sender': curr_sender,
                                          'message': curr_message}, ignore_index = True)

    return chat_df