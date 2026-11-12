**Error analysis**

The model correctly classified 295 of 400 test customers, giving 73.75% accuracy. Since 68.25% of customers stayed, predicting “Stayed” for everyone would already achieve 68.25% accuracy. For a retention team, the more important question is how many departing customers the model identified. It caught 53 of 127, giving a churn recall of 41.73%. It missed 74 customers who left and incorrectly flagged 31 who stayed.

Monthly contracts had the highest error rate (31.12%), followed by one-year contracts (22.22%) and two-year contracts (20.51%). Individual cases show why prediction was difficult. The model estimated a 78.4% churn probability for C1273, who stayed, and 41.0% for C1016, who left. In this synthetic dataset, customers with similar risk factors can have different outcomes.

**Final business recommendation**

I would rate the model’s performance as moderate. It could help a retention team prioritize customers for outreach after further validation. Before using it, I would test it on real customer records and compare decision thresholds on a separate validation set. A lower threshold could catch more customers who are likely to leave, but it would also flag more customers who would stay. I would choose the threshold based on how many customers the team can contact and the cost of missing someone who leaves.


# Customer Churn Prediction

I trained a logistic regression model to predict which customers would leave an internet service. The synthetic dataset contained 2,000 customers. I used 1,600 for training and 400 for testing, with the same churn rate in both groups.

## Five findings

1. Overall, 635 customers left the service, giving a churn rate of 31.75%.
2. Monthly contracts had the highest churn rate at 39.08%, followed by two-year contracts at 25.46% and one-year contracts at 23.70%.
3. The median monthly charge was 81.84 for customers who left and 65.29 for those who stayed.
4. Customers who left had a median tenure of 27 months, compared with 41 months for those who stayed.
5. Churn rose from 19.47% among customers who made no support calls to 58.88% among those who made five or more.

## Model results

The model correctly classified 295 of 400 test customers, giving 73.75% accuracy. It correctly identified 242 customers who stayed and 53 who left. It missed 74 customers who left and incorrectly flagged 31 who stayed.

For a retention team, the main limitation is that the model caught only 53 of the 127 customers who left, giving 41.73% churn recall. Monthly contracts were the hardest group to classify, with an error rate of 31.12%.

I also tested the model on 10 new customer profiles. At the 50% decision threshold, it flagged five as likely to leave. NEW05 had the highest estimated churn probability at 97.5%.

## Business recommendation

I would rate the model’s performance as moderate. Before using it for retention outreach, I would validate it on real customer data and check whether a lower threshold catches more departing customers at a contact volume the team can manage.