import time
import streamlit as st
from graph import Components

# Load environment variables (TAVILY_API_KEY) from .env file
from dotenv import load_dotenv
load_dotenv()

class GraphInitializer():
    def __init__(self, input_email, model_name="llama3"):
        self.input_email = input_email
        self.nodes = Components(input_email, model_name)

    def run(self):
        self.nodes.run()

def main():
    st.title("Email Response Generator")
    st.write("Enter your email below to get a reply email:")

    input_email = st.text_area("Your Email", height=200, placeholder="Paste your email here...")

    if st.button("Generate Reply"):
        if input_email:
            st.write("Processing...")
            st_time = time.time()
            graph_initializer = GraphInitializer(input_email, model_name="llama3")
            # Assuming `graph_initializer.nodes.get_reply()` gives the generated reply
            reply_email = graph_initializer.nodes.run()
            processing_time = time.time() - st_time
            st.write(f"Time taken: {processing_time:.2f} seconds")
            st.write("Reply Email:")
            st.text_area("Generated Reply", reply_email, height=200)
        else:
            st.write("Please enter an email to generate a reply.")

if __name__ == "__main__":
    main()
