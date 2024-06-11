import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import preprocessor, helper
from wordcloud import WordCloud
import seaborn as sns
from matplotlib.font_manager import FontProperties
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.sidebar.title("WattsApp Chat Analyzer")


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    #st.dataframe(df)

    # fetch unique users
    users_list = df["user"].unique().tolist()
    users_list.remove("group_notification")
    users_list.sort()
    users_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Analysis w.r.t to users", users_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, num_media, num_links = helper.fetch_stats(
            selected_user, df
        )
        
        st.title("Top Statistics📊")
        col1, col2, col3, col4 = st.columns(4,gap="small")

        with col1:
            st.header("Texts📝")
            st.title(num_messages)
        with col2:
            st.header("Words🅰")
            st.title(words)
        with col3:
            st.header("Media📷")
            st.title(num_media)
        with col4:
            st.header("Links🌐")
            st.title(num_links)

        # Timeline
        st.title("Monthly_Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color="#503f3f")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #activity map:
        st.title("Activity Map")
        col1,col2  = st.columns(2,gap="small")
        
        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig, ax = plt.subplots()
            ax.barh(busy_day.index,busy_day.values,color="#54bebe")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user,df)
            fig, ax = plt.subplots()
            ax.barh(busy_month.index,busy_month.values,color="#c80064")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)
        
        # finding the busiest users in the group
        if selected_user == "Overall":
            st.title("Busiest Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2,gap="small")

            with col1:
                ax.bar(x.index, x.values, color="#466964")
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

       
        # WordCloud
        st.title("WordCloud")
        df_wc = helper.create_word_cloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation="bilinear")
        st.pyplot(fig)


        #most common words
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color="#5e569b")
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        #Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user,df)
        col1,col2 = st.columns(2,gap="small")

        with col1:
            st.dataframe(emoji_df)

        with col2:
            font_prop = FontProperties(fname="C:\\Users\\Lenovo\\Downloads\\Segoe-UI-Emoji_FontYukle\\Segoe UI Emoji.TTF")
            #font_prop = FontProperties(fname="C:\\Users\\Lenovo\\Downloads\\Noto_Color_Emoji\\NotoColorEmoji-Regular.ttf")

            # Use the font in your plot
            plt.rcParams['font.family'] = font_prop.get_name()

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                emoji_df[1].head(),
                labels=emoji_df[0].head(),
                autopct="%.2f",
                textprops=dict(fontproperties=font_prop)
            )
            plt.setp(texts, fontproperties=font_prop)
            plt.setp(autotexts, fontproperties=font_prop)
            
            ax.set_title("Top Emojis Used", fontproperties=font_prop)
            st.pyplot(fig)
        

        # Sentiment Analysis

        def categorize_sentiment(score):
            if score >= 0.5:
                return "Positive"
            elif score <= -0.5:
                return "Negative"
            else:
                return "Neutral"
    
        st.title("Sentiment Analysis")
        avg_sentiment, sentiment_scores = helper.sentiment_analysis(selected_user, df)
        overall_sentiment = categorize_sentiment(avg_sentiment)
        st.write(f"Overall Sentiment: {overall_sentiment}  (Average Sentiment Score: {avg_sentiment:.2f})")

        fig, ax = plt.subplots()
        ax.hist(sentiment_scores, bins=20, color="grey", edgecolor='white')
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
