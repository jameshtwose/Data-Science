#%%
from atlassian import Confluence
from dotenv import load_dotenv, find_dotenv
from glob import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
#%%
load_dotenv(find_dotenv())

confluence_MV = Confluence(
    url=os.environ["atlassian_url"],
    username=os.environ["atlassian_username_tink"],
    password=os.environ["atlassian_api_token_tink"],
    api_version="cloud"
    # cloud=True
    )

confluence_DS = Confluence(
    url=os.environ["atlassian_url"],
    username=os.environ["atlassian_username_gio"],
    password=os.environ["atlassian_api_token_gio"],
    api_version="cloud"
    # cloud=True
    )

#%%
# confluence.get_all_spaces(start=0, 
#                         limit=10, 
#                         expand=None)

# %%
DS_meta_data = confluence_DS.get_all_pages_from_space(space="DS", 
start=0, 
limit=1000, status=None, expand=None, content_type='page')
# %%
id_list = [x["id"] for x in DS_meta_data]
# id_list = [1130397701, 1376845825, 1287684100]
# %%
for id in id_list:
    page_info = confluence_DS.get_page_by_id(page_id=id, expand=None, status=None, version=None)
    pdf_name = page_info["title"].replace(" ", "_")
    print(pdf_name)
    try:
        with open(f"output_pdfs/DS_pdfs/{pdf_name}.pdf", "wb") as pdf_file:
            pdf_file.write(confluence_DS.get_page_as_pdf(id))
    except:
        print("The above file had a name which is not allowed")

# %%
CT_meta_data = confluence_MV.get_all_pages_from_space(space="MV", 
start=0, 
limit=10000, status=None, expand=None, content_type='page')
# %%
id_list = [x["id"] for x in CT_meta_data][225:]
# %%
for id in id_list:
    page_info = confluence_MV.get_page_by_id(page_id=id, expand=None, status=None, version=None)
    pdf_name = page_info["title"].replace(" ", "_")
    print(pdf_name)
    try:
        with open(f"output_pdfs/CT_pdfs/{pdf_name}.pdf", "wb") as pdf_file:
            pdf_file.write(confluence_MV.get_page_as_pdf(id))
    except:
        print("The above file had a name which is not allowed")
# %%
id = id_list[30]
# confluence.get_page_as_pdf(id)
confluence_MV.export_page(page_id=id)
# %%
os.getcwd()
# %%
id_list
# %%
len(glob("output_pdfs/CT_pdfs/*"))
# %%
