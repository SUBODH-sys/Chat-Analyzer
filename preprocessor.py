import pandas as pd
import re


def preprocess(data):
    pattern = "\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s\w+\s-\s"
    #pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({"user_message": messages, "message_date": dates})

    # a function to parse the date with the correct format
    def parse_date(date_str):
        return pd.to_datetime(date_str.strip(), format="%m/%d/%y, %I:%M\u202F%p -")

    # Convert message_date type
    df["message_date"] = df["message_date"].apply(parse_date)

    df.rename(columns={"message_date": "date"}, inplace=True)

    # Separate users and messages
    users = []
    messages = []

    for i in df["user_message"]:
        entry = re.split("([\w\W]+?):\s", i)
        if entry[1:]:  # username and message exist
            users.append(entry[1])
            messages.append(entry[2])
        else:  # system message or notification
            users.append("group_notification")
            messages.append(entry[0])

    df["user"] = users
    df["message"] = messages
    df.drop(columns=["user_message"], inplace=True)

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    period = []


    for hour in df[["day_name", "hour"]]["hour"]:
          if hour == 23:
                    period.append(str(hour) + "-" + str("00"))
          elif hour == 0:
                    period.append(str("00") + "-" + str(hour + 1))
          else:
                    period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
