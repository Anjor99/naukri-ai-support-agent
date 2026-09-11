from agent.conversation import run_turn
from agent.memory import clear_conversation


def main():
    conversation_id = "task8-demo"

    # Start fresh
    clear_conversation()

    print("\n--- TURN 1 ---")
    result1 = run_turn(
        "What is the status of APP-0001?",
        conversation_id=conversation_id,
    )

    print(result1["response"])
    print("Remembered record ID:", result1.get("record_id"))

    print("\n--- TURN 2 ---")
    result2 = run_turn(
        "What is its expected salary?",
        conversation_id=conversation_id,
    )

    print(result2["response"])
    print("Remembered record ID:", result2.get("record_id"))

    print("\n--- FRESH CONVERSATION ---")
    result3 = run_turn(
        "What is its expected salary?",
        conversation_id="fresh-conversation",
    )

    print(result3["response"])
    print("Remembered record ID:", result3.get("record_id"))


if __name__ == "__main__":
    main()