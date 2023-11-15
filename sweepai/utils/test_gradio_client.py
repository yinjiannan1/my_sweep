from gradio_client import Client

with open("/Users/jiannan.yinthoughtworks.com/OTR/otr-performance/sweep/sweepai/utils/test.txt", "r") as f:
    text = f.read()

client = Client("http://47.103.63.15:50085/")
result = client.predict(
				"？",	# str in 'Instruction' Textbox component
				0.5,	# int | float (numeric value between 0 and 1)in 'Temperature' Slider component
				2048,	# int | float (numeric value between 1 and 2048)in 'Max tokens' Slider component
				api_name="/predict"
)
print(result)

