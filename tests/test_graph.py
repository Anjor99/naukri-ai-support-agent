from agent.graph import app


result = app.invoke({
    "query": "What is the status of APP-0001?"
})

print(result)