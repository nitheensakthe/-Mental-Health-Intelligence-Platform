import networkx as nx
import pandas as pd

class ContextualGraphBuilder:
    """
    Contextual Graph Construction Module for CGNIVEX.
    Builds a heterogeneous network graph containing:
    Nodes: Users, Posts, Topics, Keywords, Emotions
    Edges: User->Post, Post->Topic, Post->Keyword, User->User, Topic->Topic
    """

    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Constructs heterogeneous graph from a processed DataFrame.
        """
        self.graph.clear()

        if df.empty:
            return self.graph

        user_posts = {}

        for _, row in df.iterrows():
            post_id = f"post_{row.get('post_id', '')}"
            user_id = f"user_{row.get('user_id', '')}"
            topic = str(row.get('topic', 'General'))
            emotion = str(row.get('label', 'Neutral'))
            text = str(row.get('cleaned_text', row.get('text', '')))
            keywords = [w for w in text.split() if len(w) > 3][:4]

            # 1. Add Nodes with type attribute
            self.graph.add_node(user_id, type="User", label=user_id)
            self.graph.add_node(post_id, type="Post", label=post_id, text=text[:30])
            self.graph.add_node(f"topic_{topic}", type="Topic", label=topic)
            self.graph.add_node(f"emotion_{emotion}", type="Emotion", label=emotion)

            for kw in keywords:
                self.graph.add_node(f"kw_{kw}", type="Keyword", label=kw)

            # 2. Add Edges
            # User -> Post
            self.graph.add_edge(user_id, post_id, relation="authored")
            
            # Post -> Topic
            self.graph.add_edge(post_id, f"topic_{topic}", relation="categorized_under")
            
            # Post -> Emotion
            self.graph.add_edge(post_id, f"emotion_{emotion}", relation="expresses")

            # Post -> Keyword
            for kw in keywords:
                self.graph.add_edge(post_id, f"kw_{kw}", relation="contains_keyword")

            # Track user posts for User -> User co-interaction links
            if user_id not in user_posts:
                user_posts[user_id] = []
            user_posts[user_id].append(topic)

        # 3. Add User -> User edges (if users share discussions on same topics)
        users = list(user_posts.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u1, u2 = users[i], users[j]
                shared_topics = set(user_posts[u1]).intersection(set(user_posts[u2]))
                if shared_topics:
                    self.graph.add_edge(u1, u2, relation="co_participates", weight=len(shared_topics))

        # 4. Add Topic -> Topic edges (topics co-occurring in same community)
        topics = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Topic"]
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                if i != j:
                    self.graph.add_edge(topics[i], topics[j], relation="topic_relation", weight=1.0)

        print(f"[GraphBuilder] Constructed Graph: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")
        return self.graph

    def update_graph_dynamically(self, new_post_dict: dict) -> nx.Graph:
        """
        Dynamically updates existing graph when a new post arrives in real time.
        """
        post_id = f"post_{new_post_dict.get('post_id', 'new')}"
        user_id = f"user_{new_post_dict.get('user_id', 'anonymous')}"
        topic = str(new_post_dict.get('topic', 'General'))
        emotion = str(new_post_dict.get('emotion', 'Neutral'))
        text = str(new_post_dict.get('text', ''))
        keywords = [w for w in text.split() if len(w) > 3][:4]

        self.graph.add_node(user_id, type="User", label=user_id)
        self.graph.add_node(post_id, type="Post", label=post_id, text=text[:30])
        self.graph.add_node(f"topic_{topic}", type="Topic", label=topic)
        self.graph.add_node(f"emotion_{emotion}", type="Emotion", label=emotion)

        self.graph.add_edge(user_id, post_id, relation="authored")
        self.graph.add_edge(post_id, f"topic_{topic}", relation="categorized_under")
        self.graph.add_edge(post_id, f"emotion_{emotion}", relation="expresses")

        for kw in keywords:
            self.graph.add_node(f"kw_{kw}", type="Keyword", label=kw)
            self.graph.add_edge(post_id, f"kw_{kw}", relation="contains_keyword")

        return self.graph
