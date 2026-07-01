import sys
import dine_ai.evaluations.integration_test as it

# Expose classes for library integration compatibility
IntegrationTestRunner = it.IntegrationTestRunner
IntegrationReport = it.IntegrationReport

if __name__ == "__main__":
    runner = it.IntegrationTestRunner()
    all_passed = runner.run_all()
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)
