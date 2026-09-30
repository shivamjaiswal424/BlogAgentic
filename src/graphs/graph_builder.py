from langgraph.graph import END, START, StateGraph

from src.llms.groqllm import GroqLLM
from src.nodes.blog_node import BlogNode
from src.states.blogstate import BlogState


class GraphBuilder:
    def __init__(self,llm,research_llm=None):
        self.llm=llm
        self.research_llm=research_llm
        self.graph=StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Build a graph to generate blogs based on topic
        """


        self.blog_node_obj=BlogNode(self.llm,self.research_llm)

        ## Nodes
        self.graph.add_node("research",self.blog_node_obj.research)
        self.graph.add_node("title_creation",self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation",self.blog_node_obj.content_generation)

        ## Edges
        self.graph.add_edge(START,"title_creation")
        self.graph.add_edge("title_creation","research")
        self.graph.add_edge("research","content_generation")
        self.graph.add_node("human_approval",self.blog_node_obj.human_approval)
        self.graph.add_edge("content_generation","human_approval")
        self.graph.add_conditional_edges("human_approval",self.blog_node_obj.approval_decision,{"approved":END,"rejected":END})


        return self.graph

    def build_language_graph(self):
        """
        Build a graph for blog generation with inputs topic and language
        """
        self.blog_node_obj=BlogNode(self.llm,self.research_llm)
        ## Nodes
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("research",self.blog_node_obj.research)
        self.graph.add_node("content_generation",self.blog_node_obj.content_generation)
        self.graph.add_node("hindi_translation",lambda state: self.blog_node_obj.translation({**state, "current_language":"hindi"}))
        self.graph.add_node("french_translation",lambda state: self.blog_node_obj.translation({**state, "current_language":"french"}))
        self.graph.add_node("route",self.blog_node_obj.route)

        ## Edges &  Conditional Edge
        self.graph.add_edge(START,"title_creation")
        self.graph.add_edge("title_creation","research")
        self.graph.add_edge("research","content_generation")
        self.graph.add_edge("content_generation","route")

        self.graph.add_conditional_edges(
            "route",
            self.blog_node_obj.route_decision,
            {
                "hindi":"hindi_translation",
                "french":"french_translation"
            }
        )
        self.graph.add_node("human_approval",self.blog_node_obj.human_approval)
        self.graph.add_edge("hindi_translation","human_approval")
        self.graph.add_edge("french_translation","human_approval")
        self.graph.add_conditional_edges("human_approval",self.blog_node_obj.approval_decision,{"approved":END,"rejected":END})
        return self.graph


    def setup_graph(self, usecase,checkpointer=None):
        if usecase=="topic":
            self.build_topic_graph()

        if usecase=="language":
            self.build_language_graph()

        return self.graph.compile(checkpointer=checkpointer)


## Below code is for the langsmith/langgraph studio

groqllm=GroqLLM()
llm=groqllm.get_llm()
research_llm=groqllm.get_research_llm()

#get the graph
graph_builder=GraphBuilder(llm,research_llm)
graph=graph_builder.build_language_graph().compile()
