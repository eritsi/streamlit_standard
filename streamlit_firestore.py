import streamlit as st
from google.cloud import firestore
import json
import os

# streamlit sharingから FireStoreへアクセスするときは、secretsに置く
# https://blog.streamlit.io/streamlit-firestore-continued/
# key_dict = json.loads(st.secrets["textkey"])
# creds = service_account.Credentials.from_service_account_info(key_dict)
# db = firestore.Client(credentials=creds, project="streamlit-reddit")
prj = os.environ["GCLOUD_PROJECT"]
db = firestore.Client(project=prj)
# db = firestore.Client.from_service_account_json("firestore-key.json")

# Streamlit widgets to let a user create a new post
title = st.text_input("Post title")
url = st.text_input("Post url")
submit = st.button("Submit new post")

# Once the user has submitted, upload it to the database
if title and url and submit:
	doc_ref = db.collection("streamlit").document(title)
	doc_ref.set({
		"title": title,
		"url": url
	})

# And then render each post, using some light Markdown
posts_ref = db.collection("streamlit")
for doc in posts_ref.stream():
	post = doc.to_dict()
	title = post["title"]
	url = post["url"]

	st.subheader(f"Post: {title}")
	st.write(f":link: [{url}]({url})")