from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.states.blogstate import BlogState


class BlogNode:

    """
    A class to represent the blog node.
    """

    def __init__(self,llm,research_llm=None) -> None:
        self.llm=llm
        self.research_llm=research_llm

    def research(self,state:BlogState):
        if state.get("topic"):
            prompt=f"""You are a research assistant preparing background material for a blog writer.
            Search the web and produce a concise, factual briefing on "{state['topic']}":
            key facts, recent developments, and any statistics worth citing.
            """

            try:
                response=self.research_llm.invoke(prompt)
                sources=[]
                for tool in response.additional_kwargs.get("executed_tools") or []:
                    output=tool.get("output") or ""
                    for line in output.splitlines():
                        if line.startswith("URL:"):
                            sources.append(line.removeprefix("URL:").strip())
                            break
                return {"research_notes":response.content,"sources":sources}
            except Exception:
                fallback_prompt=f"""Give a concise factual briefing on "{state['topic']}" from what you
                already know. Note that this was NOT verified against live web search."""
                response=self.llm.invoke(fallback_prompt)
                return {"research_notes":response.content,"sources":[]}

    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if state.get("topic"):
            prompt="""
                    You are an expert blog content writer. User Markdown formatting.
                    Generate a blog title for the {topic}. This title should be creative
                    and SEO friendly.

                    """

            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)
            return {"blog":{"title":response.content}}


    def content_generation(self,state:BlogState):
        if state.get("topic"):
            system_prompt="""You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}.

            Ground your writing in this research briefing, and don't contradict it:
            {research_notes}
            """
            system_message=system_prompt.format(topic=state["topic"],research_notes=state.get("research_notes",""))
            response=self.llm.invoke(system_message)
            return {"blog" : {"title": state['blog']['title'], "content" : response.content}}

    def translation(self,state:BlogState):
        """
        Translate the content to the specified language.
        """
        translation_prompt="""
        Translate the following blog into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.
        - Respond in EXACTLY this format, with no extra commentary:

        TITLE: <translated title, one line>
        CONTENT:
        <translated content>

        ORIGINAL TITLE:
        {blog_title}

        ORIGINAL CONTENT:
        {blog_content}
        """

        messages=[
            HumanMessage(translation_prompt.format(
                current_language=state["current_language"],
                blog_title=state["blog"]["title"],
                blog_content=state["blog"]["content"],
            ))
        ]
        response=self.llm.invoke(messages,max_tokens=8000,reasoning_effort="low")
        _,_,rest=response.content.partition("TITLE:")
        title,_,content=rest.partition("CONTENT:")
        title=title.strip()
        content=content.strip()
        if not title or not content:
            return {"blog":state["blog"]}
        return {"blog":{"title":title,"content":content}}


    def route(self,state:BlogState):
        return {"current_language":state['current_language']}

    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """
        if state["current_language"]=="hindi":
            return "hindi"
        elif state["current_language"]=="french":
            return "french"
        else:
            return state['current_language']

    def human_approval(self,state:BlogState):
        decision=interrupt({"blog":state["blog"],"message":"Approve this blog for publishing?"})
        return {"approval_status":decision}

    def approval_decision(self,state:BlogState):
        if state.get("approval_status")=="approved":
            return "approved"
        return "rejected"

    