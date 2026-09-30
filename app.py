import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from langgraph.types import Command

from src.auth import verify_api_key
from src.db.postgres import (
    close_db,
    create_run,
    get_checkpointer,
    get_run,
    init_db,
    update_run_status,
)
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

@asynccontextmanager
async def lifespan(app:FastAPI):
    await init_db()
    yield
    await close_db()

app=FastAPI(lifespan=lifespan,dependencies=[Depends(verify_api_key)])


##API's

async def run_graph(run_id:str,topic:str,language:str):
    await update_run_status(run_id,"running")
    try:
        groqllm=GroqLLM()
        llm=groqllm.get_llm()
        research_llm=groqllm.get_research_llm()

        graph_builder=GraphBuilder(llm,research_llm)
        config={"configurable":{"thread_id":run_id}}

        if language:
            graph=graph_builder.setup_graph(usecase="language",checkpointer=get_checkpointer())
            result=await graph.ainvoke({"topic":topic,"current_language":language.lower()},config=config)
        else:
            graph=graph_builder.setup_graph(usecase="topic",checkpointer=get_checkpointer())
            result=await graph.ainvoke({"topic":topic},config=config)

        blog=result["__interrupt__"][0].value["blog"]
        await update_run_status(run_id,"awaiting_approval",blog["title"],blog["content"])
    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        await update_run_status(run_id,"failed")


@app.post("/runs")
async def start_run(request:Request,background_tasks:BackgroundTasks):
    data=await request.json()
    topic=data.get("topic","")
    language=data.get("language",'')

    if not topic:
        raise HTTPException(status_code=400,detail="topic is required")

    run_id=str(uuid.uuid4())
    await create_run(run_id,topic,language)
    background_tasks.add_task(run_graph,run_id,topic,language)

    return {"run_id":run_id}


@app.get("/runs/{run_id}")
async def get_run_status(run_id:str):
    row=await get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404,detail="run not found")
    return row


@app.post("/runs/{run_id}/decision")
async def submit_decision(run_id:str,request:Request):
    data=await request.json()
    decision=data.get("decision")
    if decision not in ("approved","rejected"):
        raise HTTPException(status_code=400,detail="decision must be 'approved' or 'rejected'")

    row=await get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404,detail="run not found")
    if row["status"]!="awaiting_approval":
        raise HTTPException(status_code=409,detail=f"run is '{row['status']}', not awaiting approval")

    groqllm=GroqLLM()
    llm=groqllm.get_llm()
    research_llm=groqllm.get_research_llm()

    graph_builder=GraphBuilder(llm,research_llm)
    usecase="language" if row["language"] else "topic"
    graph=graph_builder.setup_graph(usecase=usecase,checkpointer=get_checkpointer())
    config={"configurable":{"thread_id":run_id}}

    final=await graph.ainvoke(Command(resume=decision),config=config)
    await update_run_status(run_id,final["approval_status"],final["blog"]["title"],final["blog"]["content"])

    return {"run_id":run_id,"status":final["approval_status"]}


if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)
