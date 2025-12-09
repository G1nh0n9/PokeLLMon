from openai import OpenAI

def responses(**params):
    client = OpenAI()
    response = client.responses.create(
        **params
    )

    print(response)
    print(dir(response))
    return response