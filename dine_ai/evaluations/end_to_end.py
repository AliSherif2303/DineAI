import sys
import dine_ai.evaluations.end_to_end_test as e2e

if __name__ == "__main__":
    runner = e2e.EndToEndTestRunner()
    all_passed = runner.run_all()
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)
