# from pattern.en import parse
# from pattern.en import pprint
# from pattern.en import sentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import colorsys
import matplotlib.pyplot as plt


# from flair.models import TextClassifier
# from flair.data import Sentence
# import plotly.express as px
# from IPython.display import display
# df = px.data.tips()

# display(df)

# fig = px.treemap(df, path=[px.Constant("all"), 'day', 'time', 'sex'], values='total_bill')
# fig.update_traces(root_color="lightgrey")
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()

#emotion

def color_from_sentiment(sentiment):
    #if the value is 1 or -1, it will be black, adjust
    if sentiment == 1.0:
        sentiment-=0.01
    if sentiment == -1.0:
        sentiment+=0.01
    #blue
    if sentiment >= 0:
        color = colorsys.rgb_to_hls(sentiment, sentiment, 1)
    #red
    else:
        sentiment = sentiment*-1
        color = colorsys.rgb_to_hls(1, sentiment, sentiment)

    color_str = str(color[0])+" "+str(color[1])+" "+str(color[2])
    return color_str

class e():
    def __init__(self, name, parent, strength):
        self.name = name
        self.parent = parent
        self.strength = strength
        self.negative = False
        self.definition = ""

    def __str__(self):
        return str(self.name)+":"+str(self.parent)+":"+str(self.strength)+":"+str(self.negative)

    def __repr__(self):
        return self.__str__()

    def dot(self):
        color = color_from_sentiment(self.strength)
        color_string = " [fillcolor=\""+color+"\" style=filled ] "
        # print(color)

        node_string = str(self.name)+ color_string + ";"
        edge_string = str(self.parent)+" -> "+str(self.name)+";"
        if self.parent == None:
            return node_string
        else:
            return node_string + "\n" + edge_string
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
    e("loathing", "disapproval", -0.7),
    e("disappointed", "disgust", -0.5),
    e("repugnant", "disappointed", -0.8),
    e("revolted", "disappointed", -0.9),
    e("awful", "disgust", -0.7),
    e("revulsion", "awful", -0.8),
    e("detestable", "awful", -0.3),
    e("avoidance", "disgust", -0.5),
    e("aversion", "avoidance", -0.3),
    e("hesitant", "avoidance", -0.5),
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
    e("stupid", "ashamed", -0.9),
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
    e("intimate", "happy", 0.5),
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
    e("hopeful", "peaceful", 0.7),
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
    e("caring","thoughtful", 0.5),
    e("trusting", "peaceful", 0.5),
    e("secure", "trusting", 0.7),
    e("nurturing", "trusting", 0.8),
    e("thankful", "nurturing", 0.9),
    #suprise
    e("surprise", "emotion", 0.5),
    e("excited", "surprise", 0.7),
    e("energetic", "excited", 0.8),
    e("stimulating", "energetic", 0.9),
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

def build_dict_tree(emotions):
    tree = dict()
    for e in emotions:
        if e.parent in tree:
            tree[e.parent].append(e)
        else:
            tree[e.parent] = [e]
    return tree

position_dict = dict()
global_counter = 0
def in_order_traverse(tree, name):
    global global_counter
    global position_dict
    if name in tree:
        i=0
        added=False
        for e in tree[name]:
            in_order_traverse(tree, e.name)
            i=i+1
            if i > len(tree[name])/2 and not added:
                position_dict[name] = global_counter
                global_counter= global_counter + 1
                added=True
    else:
        position_dict[name] = global_counter
        global_counter= global_counter + 1

def traverse_children(tree, name):
    children = []
    if name in tree:
        for e in tree[name]:
            children.append(e.name)
            children.extend(traverse_children(tree, e.name))
    return children

def get_children(emotions, name):
    tree = build_dict_tree(emotions)
    children = traverse_children(tree, name)
    return children

def get_level(emotions, name):
    edict = emotion_dict(emotions)
    level = 0
    while edict[name].parent != "emotion":
        name = edict[name].parent
        level = level + 1
    return level

def get_chain_csv(emotions, name):
    edict = emotion_dict(emotions)
    chain = []
    while edict[name].parent != "emotion":
        chain = chain + edict[name].parent + ","
        name = edict[name].parent
    return chain

def emotion_csv(emotions):
    for e in emotions:
        print(get_chain_csv(emotions, e.name))

def recursive_number_of_children(emotions, name):
    edict = emotion_dict(emotions)
    if name in edict:
        count = 0
        for e in edict[name].children:
            count = count + recursive_number_of_children(emotions, e)
        return count + 1
    else:
        return 0

def scatter2(emotions):
    tree = build_dict_tree(emotions)
    in_order_traverse(tree, "emotion")
    # print(position_dict)
    #all positive
    i=0
    for e in emotions:
        if e.strength < 0:
            emotions[i].strength = abs(emotions[i].strength)
        print(e.name + " " + str(position_dict[e.name]))
        i=i+1

    x = []
    y = []
    sizes=[]
    labels = []
    i=0
    for e in emotions:
        x.append(e.strength)
        y.append(position_dict[e.name])
        #todo start here I was trying to get node sizes for each of the number of childern a node has
        sizes.append(recursive_number_of_children(emotions, e.name))
        labels.append(e.name)
        i=i+1

    fig, ax = plt.subplots(1,1,figsize=(15,20))
    ax.scatter(x, y, s=sizes)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]), fontsize=15)
    
    edges = create_edges(emotions)
    for e in edges:
        level = get_level(emotions, e[2])
        ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='black', linestyle=':', linewidth=1)
    plt.savefig('emotion_scatter2.pdf')

def emotion_dict(emotions):
    edict = dict()
    for e in emotions:
        edict[e.name] = e
    return edict

def create_edges(emotions):
    edict = emotion_dict(emotions) 
    edges=[]
    for e in emotions:
        if e.name != "emotion":
            print(e)
            parent = edict[e.parent]
            child = edict[e.name]
            edges.append(((parent.strength, position_dict[e.parent]), (child.strength, position_dict[e.name]), e.name))
    return edges

# classifier = TextClassifier.load('en-sentiment')
def sentiment_analyze_emotions(emotions):
    analyzer = SentimentIntensityAnalyzer()
    for e in emotions:
        name = ("I feel " + e.name + " right now.")
        vs = analyzer.polarity_scores(name)
        # print("{:-<65} {}".format(e.name, str(vs)))
        # print(e.name + " " + str(vs['compound']))
        e.strength = vs['compound']
        if e.strength < 0:
            e.negative = True

    emotions = normalize(emotions)
    # emotions = filter(emotions,0.1)
    return emotions

def digraph(emotions):
    f = open("emotions.dot", "w")
    f.write("digraph emotions {\n")
    for e in emotions:
        f.write("\t" +e.dot() + "\n")
    f.write("}\n")
    f.close()

def subgraph_generator(emotions, name):
    children = get_children(emotions, name)
    child_str = ""
    i=0
    for c in children:
        child_str += c
        if i < len(children)-1:
            child_str = child_str + " -> "
        i=i+1
    return child_str

def nested_digraph_root(emotions):
    #todo start here https://stackoverflow.com/questions/7777722/top-down-subgraphs-left-right-inside-subgraphs
    f = open("emotions_nested.dot", "w")
    f.write("digraph emotions {\n")
    f.write("rankdir=\"TB\"\n")
    # f.write("rankdir=\"LR\"\n")
    # f.write("emotions\n")

    nested_digraph = digraph_nested(emotions, "emotion")

    f.write(nested_digraph)
    # f.write("}\n")#emotion subcluster

    f.write("}\n")
    f.close()

def digraph_nested(emotions, name):
    # edict = emotion_dict(emotions)
    tree = build_dict_tree(emotions)
    subgraph = subgraph_generator(emotions, name)

    ret_val = ""

    if subgraph == "":
        return "\n"

    ret_val += "subgraph cluster_" + name + " {\n"
    ret_val += "newrank=true\n"
    # ret_val += "rankdir=LR\n"
    ret_val += "rankdir=\"LR\"\n"
    ret_val += "edge [style=invis]"

    if name in tree:
        for children in tree[name]:
            ret_val += digraph_nested(emotions, children.name)
    
    ret_val += subgraph + "\n"
    ret_val += "}\n"
    return ret_val

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

def filter(emotions, threshold):
    filtered = []
    for e in emotions:
        if abs(e.strength) > threshold:
            filtered.append(e)
    return filtered

def scatter(emotions):
    x = []
    y = []
    labels = []
    i=0
    for e in emotions:
        x.append(e.strength)
        y.append(i)
        labels.append(e.name)
        i=i+1

    fig, ax = plt.subplots(1,1,figsize=(15,20))
    ax.scatter(x, y)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))
    plt.savefig('emotion_scatter.pdf')

def nested_tree_graph():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #We will generate a small and simple data frame for plotting the treemaps so that it is easier to compare the syntax and look of these plots in different libraries.
    colors=['#fae588','#f79d65','#f9dc5c','#e8ac65','#e76f51','#ef233c','#b7094c'] #color palette
    data = {'labels': ["A","B","C","D","E","F","G"],
            'values':[10,20,20,35,10,25,45]}
    df = pd.DataFrame(data)
    print(df) #print the dataframe


    import squarify
    sns.set_style(style="whitegrid") # set seaborn plot style
    sizes= df["values"].values# proportions of the categories
    label=df["labels"]
    squarify.plot(sizes=sizes, label=label, alpha=0.6,color=colors).set(title='Treemap with Squarify')
    plt.axis('off')
    plt.show()

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
        # values.append(abs(int(e.strength * 100))+1 )

    # print(values)
    # fig = px.treemap(
    #     labels = names,
    #     parents = parents,
    #     values = values,
    #     branchvalues= 'total',
    # )
    # # fig.update_traces(root_color="lightgrey")
    # # fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    # fig.show()

    # values = [0, 11, 12, 13, 14, 15, 20, 30]
    # labels = ["container", "A1", "A2", "A3", "A4", "A5", "B1", "B2"]
    # parents = ["", "container", "A1", "A2", "A3", "A4", "container", "B1"]

    fig = go.Figure(go.Treemap(
        labels = names,
        # values = values,
        parents = parents,
        # color = intensity,
        # color_continuous_scale = 'RdBu',
        # color_continuous_midpoint = 0,
        sort=True,
        marker=dict(
            colors=intensity,
            # colorscale='RdBu',
            # colorscale='Picnic',
            colorscale='RdYlBu',
            cmid=0),
        text=definitions,

        # marker_colors = ["pink", "royalblue", "lightgray", "purple", 
        #                 "cyan", "lightgray", "lightblue", "lightgreen"]
    ))

    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    # fig.show()
    fig.write_html("emotion_tree.html")

# nested_tree_graph()

def get_remote_definition(emotion):
        import requests
        definition=""
        x = requests.get('https://api.dictionaryapi.dev/api/v2/entries/en/' + emotion)
        if x.status_code == 200:
            definition = x.json()[0]['meanings'][0]['definitions'][0]['definition']
        else:
            definition = "No definition found"


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
    
    with open(filename, 'wb') as handle:
        pickle.dump(definitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

emotions = sentiment_analyze_emotions(emotions)
emotions.sort(key=lambda x: x.strength, reverse=True)
# emotion_csv(emotions)

get_definitions(emotions)
# exit(1)

plotly_tree_graph(emotions)
#todo I can do nested subgraphs using this syntax in dot
#https://stackoverflow.com/questions/69399948/graphviz-nested-subgraph-orientation
#this enables me to sort, and to use "tremap" like layouts for a decent layout.

# digraph(emotions)
# digraph_nested(emotions, "happy")
nested_digraph_root(emotions)
# scatter(emotions)
# scatter2(emotions)



