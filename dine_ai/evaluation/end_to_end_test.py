import sys
import dine_ai.evaluations.end_to_end_test as e2e

# Expose classes for library integration compatibility
EndToEndTestRunner = e2e.EndToEndTestRunner
EndToEndReport = e2e.EndToEndReport

if __name__ == "__main__":
    runner = e2e.EndToEndTestRunner()
    all_passed = runner.run_all()
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)
