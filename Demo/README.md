# Customer-support demo

NeuralOps evaluates two policies for selecting a support answer. Version A recognizes one primary keyword per policy. Version B also recognizes phrases like “money back,” “locked out,” and “unsubscribe.” Both select from the same six policy answers.

This is a reproducible, synthetic benchmark of the release workflow with deterministic responders and a term-matching rubric. It measures behavior on the checked-in 18 cases. A live model comparison would use a held-out set and a separate human rubric.

## Results

| Metric | A · keywords | B · keywords and aliases |
| --- | ---: | ---: |
| Cases containing every expected term | 7 / 18 | 18 / 18 |
| Mean rubric score out of 10 | 6.28 | 9.40 |

The observed paired mean difference was **+3.12 points** for B. The 95% paired t interval was **[+1.84, +4.39]** and the two-sided p-value was **0.0000771**. Treat the interval and p-value as examples of the calculation on this curated set. Several cases share policy topics and the aliases were designed for the same domain, so the benchmark provides limited evidence about generalization.

Reproduce the numbers from the repository root with `python -m neuralops.demo --benchmark`. The fixed questions, expected terms, and scoring rules are in the [demo source](../neuralops/demo.py).

### Manual spot check

| Case | Review |
| --- | --- |
| “Can I get my money back?” | A escalates. B correctly states the refund window and original payment method. |
| “Where can I see my receipt?” | A escalates. B gives the Billing → Invoices path and PDF download. |
| “How do I reset my password?” | Both give the reset workflow and mention the email link. |
| “Can I get a refund on day 31 after delivery?” | The rubric gives both versions full term credit. Both state the 30-day policy but leave the case unanswered. A complete reply would say that day 31 falls outside the standard window. |

For a live experiment, record the data and model versions, lock the test cases before collecting results, and have reviewers score a sample against specific expected answers. Inspect disagreements like the day-31 example before relying on the judge's recommendation.

## Record the walkthrough

1. Show the two prompt versions and their diff in **Prompt Manager**.
2. Open the completed benchmark in **Experiments**. Show the paired result, then choose “Can I get my money back?” in the trial selector to show the side-by-side answers.
3. Activate `2.0.0`, then roll back to `1.0.0`. Capture the active version changing.
4. Save the video as `Demo/demo.mp4`. In the [root README](../README.md), change the link around `Demo/Thumbnail.png` from `#run-the-demo` to `Demo/demo.mp4`.
