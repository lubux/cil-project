from CNNTwitter import TweetClassifier


with TweetClassifier("./model/model5/", "./model/model5/temp.p") as predictor:
    while True:
        tweet = input("Type in your tweet:\n")
        tweet = tweet.lower()
        if tweet == "quit" or tweet == "exit":
            break
        pred = predictor.classify_tweet(tweet)
        if pred == 0:
            print("#############################")
            print("# This tweet is positive :) #")
            print("#############################")
        else:
            print("#############################")
            print("# This tweet is negative :( #")
            print("#############################")
