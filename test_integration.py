from src.main_system import CyberbullyingSystem


def run_integration():
    system = CyberbullyingSystem()
    examples = [
        ("I will kill you", True),
        ("I will not kill you", False),
        ("You are an idiot", True),
        ("You are not an idiot", False),
        ("You're a great person", False),
        ("I don't kill you", False),
    ]

    for text, expect_bullying in examples:
        res = system.analyze(text)
        is_bullying = res['is_bullying']
        print(f"{text!r} -> is_bullying={is_bullying}, severity={res['severity']}")
        assert is_bullying == expect_bullying

    print('Integration tests passed')


if __name__ == '__main__':
    run_integration()
