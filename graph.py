from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List
import pprint

from llm_utils import LLMUtils
from utils import Utils

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_email: email
        email_category: email category
        draft_email: LLM generation
        final_email: LLM generation
        research_info: list of documents
        info_needed: whether to add search info
        num_steps: number of steps
    """
    initial_email : str
    email_category : str
    draft_email : str
    final_email : str
    research_info : List[str]
    info_needed : bool
    num_steps : int
    draft_email_feedback : dict


class Components:
    def __init__(self, input_email,model_name="llama3"):
        self.input_email = input_email
        self.workflow = StateGraph(GraphState)
        self.llm_utils = LLMUtils(model_name=model_name)
        self.utils = Utils()

    def categorize_email(self, state):
        """take the initial email and categorize it"""
        print("---CATEGORIZING INITIAL EMAIL---")
        initial_email = state['initial_email']
        num_steps = int(state['num_steps'])
        num_steps += 1

        email_category = self.llm_utils.get_email_category_generator(initial_email).invoke({"initial_email": initial_email})
        print(email_category)
        # save to local disk
        self.utils.write_markdown_file(email_category, "email_category")

        return {"email_category": email_category, "num_steps":num_steps}

    def research_info_search(self, state):
        print("---RESEARCH INFO SEARCHING---")
        initial_email = state["initial_email"]
        email_category = state["email_category"]
        # research_info = state["research_info"]
        num_steps = state['num_steps']
        num_steps += 1

        # Web search
        keywords = self.llm_utils.get_search_keyword_chain(initial_email, email_category).invoke({"initial_email": initial_email,
                                                "email_category": email_category })
        keywords = keywords['keywords']
        # print(keywords)
        full_searches = []
        for keyword in keywords[:1]:
            print(keyword)
            temp_docs = self.utils.get_web_search_tool(number_of_results=1).invoke({"query": keyword})
            web_results = "\n".join([d["content"] for d in temp_docs])
            web_results = Document(page_content=web_results)
            if full_searches is not None:
                full_searches.append(web_results)
            else:
                full_searches = [web_results]
        print(full_searches)
        print(type(full_searches))
        # write_markdown_file(full_searches, "research_info")
        return {"research_info": full_searches, "num_steps":num_steps}
    
    def draft_email_writer(self, state):
        print("---DRAFT EMAIL WRITER---")
        ## Get the state
        initial_email = state["initial_email"]
        email_category = state["email_category"]
        research_info = state["research_info"]
        num_steps = state['num_steps']
        num_steps += 1

        # Generate draft email
        draft_email = self.llm_utils.get_draft_writer_chain(initial_email, email_category, research_info).invoke({"initial_email": initial_email,
                                        "email_category": email_category,
                                        "research_info":research_info})
        # print(draft_email)
        # print(type(draft_email))

        email_draft = draft_email['email_draft']
        self.utils.write_markdown_file(email_draft, "draft_email")

        return {"draft_email": email_draft, "num_steps":num_steps}
    
    def analyze_draft_email(self, state):
        print("---DRAFT EMAIL ANALYZER---")
        ## Get the state
        initial_email = state["initial_email"]
        email_category = state["email_category"]
        draft_email = state["draft_email"]
        research_info = state["research_info"]
        num_steps = state['num_steps']
        num_steps += 1

        # Generate draft email
        draft_email_feedback = self.llm_utils.get_draft_analysis_chain(initial_email, email_category, research_info, draft_email).invoke({"initial_email": initial_email,
                                                    "email_category": email_category,
                                                    "research_info":research_info,
                                                    "draft_email":draft_email}
                                                )
        # print(draft_email)
        # print(type(draft_email))

        self.utils.write_markdown_file(str(draft_email_feedback), "draft_email_feedback")
        return {"draft_email_feedback": draft_email_feedback, "num_steps":num_steps}
    
    def rewrite_email(self, state):
        print("---ReWRITE EMAIL ---")
        ## Get the state
        initial_email = state["initial_email"]
        email_category = state["email_category"]
        draft_email = state["draft_email"]
        research_info = state["research_info"]
        draft_email_feedback = state["draft_email_feedback"]
        num_steps = state['num_steps']
        num_steps += 1

        # Generate draft email
        final_email = self.llm_utils.get_rewrite_email_chain(initial_email, email_category, research_info, draft_email_feedback, draft_email).invoke().invoke({"initial_email": initial_email,
                                                    "email_category": email_category,
                                                    "research_info":research_info,
                                                    "draft_email":draft_email,
                                                    "email_analysis": draft_email_feedback}
                                                )

        self.utils.write_markdown_file(str(final_email), "final_email")
        return {"final_email": final_email['final_email'], "num_steps":num_steps}
    
    def no_rewrite(self, state):
        print("---NO REWRITE EMAIL ---")
        ## Get the state
        draft_email = state["draft_email"]
        num_steps = state['num_steps']
        num_steps += 1

        self.utils.write_markdown_file(str(draft_email), "final_email")
        return {"final_email": draft_email, "num_steps":num_steps}
    
    def state_printer(self, state):
        """print the state"""
        print("---STATE PRINTER---")
        print(f"Initial Email: {state['initial_email']} \n" )
        print(f"Email Category: {state['email_category']} \n")
        print(f"Draft Email: {state['draft_email']} \n" )
        print(f"Final Email: {state['final_email']} \n" )
        print(f"Research Info: {state['research_info']} \n")
        print(f"Info Needed: {state['info_needed']} \n")
        print(f"Num Steps: {state['num_steps']} \n")
        return
    
    def route_to_research(self, state):
        """
        Route email to web search or not.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """

        print("---ROUTE TO RESEARCH---")
        initial_email = state["initial_email"]
        email_category = state["email_category"]


        router = self.llm_utils.get_research_router(initial_email, email_category).invoke({"initial_email": initial_email,"email_category":email_category })
        print(router)
        # print(type(router))
        print(router['router_decision'])
        if router['router_decision'] == 'research_info':
            print("---ROUTE EMAIL TO RESEARCH INFO---")
            return "research_info"
        elif router['router_decision'] == 'draft_email':
            print("---ROUTE EMAIL TO DRAFT EMAIL---")
            return "draft_email"
    
    def route_to_rewrite(self, state):
        print("---ROUTE TO REWRITE---")
        initial_email = state["initial_email"]
        email_category = state["email_category"]
        draft_email = state["draft_email"]
        # research_info = state["research_info"]

        # draft_email = "Yo we can't help you, best regards Sarah"

        router = self.llm_utils.get_rewrite_router(initial_email, email_category, draft_email).invoke({"initial_email": initial_email,
                                        "email_category":email_category,
                                        "draft_email":draft_email}
                                    )
        print(router)
        print(router['router_decision'])
        if router['router_decision'] == 'rewrite':
            print("---ROUTE TO ANALYSIS - REWRITE---")
            return "rewrite"
        
        elif router['router_decision'] == 'no_rewrite':
            print("---ROUTE EMAIL TO FINAL EMAIL---")
            return "no_rewrite"
        
    def create_graph(self):

        # Define the nodes
        self.workflow.add_node("categorize_email", self.categorize_email) # categorize email
        self.workflow.add_node("research_info_search", self.research_info_search) # web search
        self.workflow.add_node("state_printer", self.state_printer)
        self.workflow.add_node("draft_email_writer", self.draft_email_writer)
        self.workflow.add_node("analyze_draft_email", self.analyze_draft_email)
        self.workflow.add_node("rewrite_email", self.rewrite_email)
        self.workflow.add_node("no_rewrite", self.no_rewrite)

        # Define the edges
        self.workflow.set_entry_point("categorize_email")

        self.workflow.add_conditional_edges(
            "categorize_email",
            self.route_to_research,
            {
                "research_info": "research_info_search",
                "draft_email": "draft_email_writer",
            },
        )
        self.workflow.add_edge("research_info_search", "draft_email_writer")


        self.workflow.add_conditional_edges(
            "draft_email_writer",
            self.route_to_rewrite,
            {
                "rewrite": "analyze_draft_email",
                "no_rewrite": "no_rewrite",
            },
        )
        self.workflow.add_edge("analyze_draft_email", "rewrite_email")
        self.workflow.add_edge("no_rewrite", "state_printer")
        self.workflow.add_edge("rewrite_email", "state_printer")
        self.workflow.add_edge("state_printer", END)


    def run(self):
        self.create_graph()
        # Compile
        app = self.workflow.compile()
        # run the agent
        inputs = {"initial_email": self.input_email,"research_info": None, "num_steps":0}
        # for output in app.stream(inputs):
        #     for key, value in output.items():
        #         pprint.pprint(f"Finished running: {key}:")
        output = app.invoke(inputs)
        return output['final_email']

        
    