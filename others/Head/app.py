import random
import time

from flask import Flask, request, jsonify
from entity.Trace import Trace
import asyncio
app = Flask(__name__)

method = 'random'

async def sample(trace):
    st = time.time()
    # sample with a fixed rate
    r = random.random()
    if r < 0.1:
        decision = True
    else:
        decision = False
    et = time.time()
    time.sleep(2)
    return



@app.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    trace = Trace.deserialize(data['trace'])
    print(trace.traceID)
    # loop = asyncio.new_event_loop()
    # loop.create_task(sample(trace))
    asyncio.run(sample(trace))

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(port=5000)