import os
from langchain_community.tools.tavily_search import TavilySearchResults
class Utils:
    def __init__(self):
        pass
    
    def write_markdown_file(self, content, filename):
        foldername = "local_disk"
        os.makedirs(foldername, exist_ok=True)
        path = os.path.join(foldername, filename)
        
        with open(f"{path}.md", "w") as f:
            f.write(content)

    def get_web_search_tool(self, number_of_results=1):
        web_search_tool = TavilySearchResults(k=number_of_results)
        return web_search_tool
