from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from matplotlib.font_manager import FontProperties
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    # number of messages
    num_messages = df.shape[0]

    # number of words
    words = []
    for i in df["message"]:
        words.extend(i.split())

    # fetch number of media messages
    num_media = df[df["message"] == "<Media omitted>\n"].shape[0]

    # fetch number of links
    links = []
    extractor = URLExtract()
    for i in df["message"]:
        links.extend(extractor.find_urls(i))

    return num_messages, len(words), num_media, len(links)


def most_busy_users(df):
    x = df["user"].value_counts().head()
    df = (
        round((df["user"].value_counts() / df.shape[0]) * 100, 2)
        .reset_index()
        .rename(columns={"user": "name"})
    )
    return x, df


def create_word_cloud(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    df_wc = wc.generate(df["message"].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for i in temp['message']:
        for word in i.lower().split():
                if word not in stop_words:
                    words.append(word)

    return_df = pd.DataFrame(Counter(words).most_common(20))
    return return_df


def emoji_helper(selected_user,df):
    if selected_user!= "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for i in df['message']:
        emojis.extend([c for c in i if emoji.is_emoji(c)])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user,df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()

    time  = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def week_activity_map(selected_user,df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user,df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user,df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    user_heatmap = df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)
    return user_heatmap


def sentiment_analysis(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Calculate average sentiment score
    avg_sentiment = df['sentiment'].mean()

    return avg_sentiment, df['sentiment']