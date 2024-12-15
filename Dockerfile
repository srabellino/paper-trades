FROM public.ecr.aws/lambda/python:3.11

# Install required libraries
RUN pip install yfinance scipy alpaca-py boto3

# Copy your Lambda function code
COPY main_efficient_lambda.py ${LAMBDA_TASK_ROOT}

# Set the Lambda function handler
CMD ["main_efficient_lambda.lambda_handler"]