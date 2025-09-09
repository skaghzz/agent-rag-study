from rag_agentic.graph import build_graph

graph = build_graph()

# for chunk in graph.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             }
#         ]
#     }
# ):
#     for node, update in chunk.items():
#         print("Update from node", node)
#         update["messages"][-1].pretty_print()
#         print("\n\n")

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Lilian Weng은 reward hacking에 대해 무엇을 말하나요?",
            }
        ]
    }
)
print(result)