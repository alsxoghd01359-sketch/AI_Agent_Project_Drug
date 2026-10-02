"""터미널에서 직접 대화해보는 실행 스크립트.

python src/chat_cli.py 로 실행하고, "종료"/"exit"/"quit"을 입력하면 끝난다.
같은 세션 안에서는 orchestrator.Conversation이 대화 기록을 유지하므로
"다른 약 복용 중인가요?" 같은 되묻기에 이어서 답할 수 있다.
"""
from orchestrator import Conversation, warmup

_EXIT_WORDS = {"종료", "exit", "quit", "그만"}


def main():
    print("AI 복약 정보 도우미입니다. 데이터를 불러오는 중입니다...")
    warmup()
    print("약에 대해 궁금한 점을 물어보세요. (종료: 종료/exit/quit)")
    conversation = Conversation()

    while True:
        try:
            user_input = input("\n[나] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input in _EXIT_WORDS:
            print("대화를 종료합니다.")
            break

        reply = conversation.send(user_input)
        print(f"\n[도우미] {reply}")


if __name__ == "__main__":
    main()
