from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import colorsys
import matplotlib.pyplot as plt

class e():
    def __init__(self, name, parent, strength):
        self.name = name
        self.parent = parent
        self.strength = strength
        self.negative = False
        self.definition = ""

    def __str__(self): 

        return str(self.name)+":"+str(self.parent)+":"+str(self.strength)+":"+str(self.negative) \
        + ":" + str(self.definition)

    def __repr__(self):
        return self.__str__()

#used the following for sentiment analysis
#https://monkeylearn.com/sentiment-analysis-online/
emotions = [
    e("emotion", "", 0),
    #anger
    e("anger", "emotion", -.649),
    e("hurt", "anger", -.852),
    e("embarrassed", "hurt", -.902),
    e("devastated","hurt", -.939),
    e("agonized", "hurt", -.960),
    e("offended", "hurt", -.5),
    e("wounded", "hurt", -.7),
    e("threatened", "anger", -.696),
    e("insecure", "threatened", -.960),
    e("jealous", "threatened", -0.5),
    e("hateful","anger", -0.9),
    e("resentful", "hateful", -0.5),
    e("violated", "hateful", -0.7),
    e("mad", "anger", -0.6),
    e("furious", "mad", -0.9),
    e("enraged", "mad", -1.0),
    e("aggressive", "anger", -0.5),
    e("provoked", "aggressive", -0.4),
    e("hostile", "aggressive", -0.7),
    e("frustrated", "anger", -0.5),
    e("infuriated", "frustrated", -0.7),
    e("irritated", "frustrated", -0.3),
    e("distant", "anger", -0.2),
    e("withdrawn", "distant", -0.3),
    e("suspicious", "distant", -0.5),
    e("aloof", "distant", -0.7),
    e("critical", "anger", -0.5),
    e("skeptical", "critical", -0.3),
    e("sarcastic", "critical", -0.5),
    #disgust
    e("disgust", "emotion", -0.5),
    e("disapproval", "disgust", -0.5),
    e("judgemental", "disapproval", -0.4),
    e("loathsome", "disapproval", -0.7),
    e("disappointed", "disgust", -0.5),
    e("repugnant", "disappointed", -0.8),
    e("revolted", "disappointed", -0.9),
    e("awful", "disgust", -0.7),
    e("revulsion", "awful", -0.8),
    e("detestable", "awful", -0.3),
    # e("avoidance", "fear", -0.5),
    e("avoidance", "disgust", -0.5), #todo also belongs under fear, consider datframe input
    e("aversion", "avoidance", -0.3),
    e("hesitant", "avoidance", -0.5),
    e("smothered", "avoidance", -0.7),
    #sad
    e("sad", "emotion", -0.5),
    e("guilty", "sad", -0.5),
    e("regretful", "guilty", -0.7),
    e("rueful", "guilty", -0.5),
    e("sorry", "guilty", -0.3),
    e("remorseful", "sorry", -0.7),
    e("apologetic", "sorry", -0.5),
    e("repentant", "sorry", -0.8),
    e("ashamed", "guilty", -0.8),
    # e("stupid", "ashamed", -0.9),
    e("ignored", "abandoned", -0.7),
    e("victimized", "abandoned", -0.8),
    e("despair", "sad", -0.5),
    e("powerless", "despair", -0.7),
    e("vulnerable", "despair", -0.8),
    e("depressed", "sad", -0.7),
    e("empty", "depressed", -0.9),
    e("lonely", "sad", -0.5),
    e("abandoned", "lonely", -0.7),
    e("isolated", "lonely", -0.8),
    e("bored", "sad", -0.5),
    e("apathetic", "bored", -0.7),
    e("indifferent", "bored", -0.8),
    e("tired", "sad", -0.2),
    e("sleepy", "tired", -0.3),
    e("exhausted", "tired", -0.5),
    #happy
    e("happy", "emotion", 0.5),
    e("optimistic", "happy", 0.7),
    e("inspired", "optimistic", 0.8),
    e("open", "optimistic", 0.9),
    e("intimate", "happy", 0.5), #todo put under delusion
    e("playful", "intimate", 0.7),
    e("sensitive", "intimate", 0.8),
    e("fascinating", "happy", 0.5),
    e("accepted", "happy", 0.5),
    e("fulfilled", "accepted", 0.7),
    e("respected", "accepted", 0.8),
    e("proud", "happy", 0.5),
    e("confident", "proud", 0.7),
    e("arrogant", "proud", 0.8),
    e("superior", "proud", 0.9),
    e("justified", "proud", 1.0),
    e("validated", "proud", 1.0),
    e("interested", "happy", 0.5),
    e("inquisitive", "interested", 0.7),
    e("amused", "interested", 0.8),
    e("joyful", "happy", 0.5),
    e("ecstatic", "joyful", 0.7),
    e("thrilled", "joyful", 0.9),
    e("blissful", "joyful", 1.0),
    e("cheerful", "happy", 0.5),
    e("creative", "happy", 0.5),
    # e("playful", "creative", 0.7),
    #powerful
    e("powerful", "emotion", 0.5),
    e("aware", "powerful", 0.7),
    e("provocative", "powerful", 0.7),
    e("appreciated", "powerful", 0.7),
    e("valuable", "appreciated", 0.8),
    e("important", "powerful", 0.7),
    e("discerning", "important", 0.8),
    # e("faithful", "powerful", 0.7),
    e("strong", "powerful", 0.7),
    e("invincible", "powerful", 0.8),
    e("free", "powerful", 0.5),
    e("liberated", "free", 0.8),
    #peaceful
    e("peaceful", "emotion", 0.5),
    e("hopeful", "despair", 0.7),
    e("loving", "peaceful", 0.8),
    e("serene", "loving", 0.5),
    e("content", "peaceful", 0.9),
    e("relaxed", "content", 0.9),
    e("calm", "content", 0.9),
    e("restful", "content", 0.9),
    e("tranquil", "content", 0.9),
    e("thoughtful", "peaceful", 0.5),
    e("pensive", "thoughtful", 0.7),
    e("contemplative", "thoughtful", 0.8),
    e("meditative", "thoughtful", 0.9),
    # e("caring","thoughtful", 0.5),
    e("trusting", "peaceful", 0.5), #todo also under submissive
    e("secure", "trusting", 0.7),
    # e("nurturing", "trusting", 0.8),
    e("thankful", "content", 0.9),
    #suprise
    e("surprise", "emotion", 0.5),
    e("excited", "happy", 0.7),
    e("energetic", "excited", 0.8),
    # e("stimulating", "energetic", 0.9),
    e("eager", "excited", 0.9),
    e("daring", "excited", 0.6),
    e("amazed", "surprise", 0.5),
    e("awe", "amazed", 0.7),
    e("astonished", "amazed", 0.8),
    e("confused", "surprise", -0.5),
    e("perplexed", "confused", -0.7),
    e("disillusioned", "confused", -0.8),
    e("startled", "surprise", -0.5),
    e("dismayed", "startled", -0.7),
    e("shocked", "startled", -0.8),
    #fear
    e("fear", "emotion", -0.5),
    e("shy", "fear", -0.5),
    e("bashful", "shy", -0.7),
    e("demure", "shy", -0.8),
    e("modest", "shy", -0.9),
    e("quiet", "shy", -0.5),
    e("timid", "shy", -0.7),
    e("scared", "fear", -0.5),
    e("cautious", "scared", -0.7),
    e("terrified", "scared", -0.7),
    e("frightened", "scared", -0.8),
    e("anxious", "fear", -0.5),
    e("overwhelmed", "anxious", -0.7),
    e("worried", "anxious", -0.8),
    # e("insecure", "fear", -0.5),
    # e("inadequate", "insecure", -0.7),
    e("inferior", "insecure", -0.8),
    e("submissive", "fear", -0.5),
    e("worthless", "submissive", -0.7),
    e("insignificant", "submissive", -0.8),
    e("rejected", "fear", -0.5),
    e("inadequate", "rejected", -0.7),
    e("alienated", "rejected", -0.8),
    e("discouraged", "rejected", -0.9),
    e("humiliated", "fear", -0.5),
    e("disrespected", "humiliated", -0.7),
    e("ridiculed", "humiliated", -0.8),
]


# classifier = TextClassifier.load('en-sentiment')
def sentiment_analyze_emotions(emotions):
    analyzer = SentimentIntensityAnalyzer()
    for e in emotions:
        name = ("I feel deeply " + e.name + " at the moment.")

        vs = analyzer.polarity_scores(name)
        e.strength = vs['compound']
        if e.strength < 0:
            e.negative = True

        if e.strength == 0:
            name += e.definition
            vs = analyzer.polarity_scores(name)
            e.strength = vs['compound']
            e.strength = e.strength * 0.5
            if e.strength < 0:
                e.negative = True

    emotions = normalize(emotions)
    return emotions

def print_emotions(emotions):
    for e in emotions:
        print(e)

def normalize(emotions):
    values = []
    for e in emotions:
        s = e.strength
        if s < 0:
            s = s * -1  
        values.append(s)

    x_norm = (values - np.min(values)) / (np.max(values) - np.min(values))

    i=0
    for e in emotions:
        emotions[i].strength = x_norm[i]
        if e.negative:
            emotions[i].strength = emotions[i].strength * -1
        i = i + 1
    return emotions

#https://plotly.com/python/treemaps/
#this page describes how to make a treemap with plotly
def plotly_tree_graph(emotions):
    import plotly.express as px
    import plotly.graph_objects as go

    names = []
    parents = []
    values = []
    intensity = []
    definitions = []
    for e in emotions:
        names.append(e.name)
        parents.append(e.parent)
        values.append(1000 + e.strength)
        intensity.append(e.strength)
        definitions.append(e.definition)

    fig = go.Figure(go.Treemap(
        labels = names,
        parents = parents,
        sort=True,
        marker=dict(
            colors=intensity,
            colorscale='RdYlBu',
            cmid=0),
        textinfo="label",
        text=definitions,
    ))

    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.write_html("index.html")

def get_remote_definition(emotion):
        import requests
        definition=""
        x = requests.get('https://api.dictionaryapi.dev/api/v2/entries/en/' + emotion)
        if x.status_code == 200:
            definition = x.json()[0]['meanings'][0]['definitions'][0]['definition']
        else:
            definition = "No definition found"
        print(emotion +":" + definition)
        return definition

def get_definitions(emotions):
    import pickle
    import os.path

    filename="definitions.pickle"
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            definitions = pickle.load(handle)
    else:
        definitions = dict()
    
    for e in emotions:
        if e.name in definitions:
            e.definition = definitions[e.name]
        else:
            e.definition = get_remote_definition(e.name)
            definitions[e.name] = e.definition

    print(definitions)
    
    with open(filename, 'wb') as handle:
        pickle.dump(definitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emotions

emotions = get_definitions(emotions)
emotions = sentiment_analyze_emotions(emotions)
print_emotions(emotions)
# emotions.sort(key=lambda x: x.strength, reverse=True)

plotly_tree_graph(emotions)