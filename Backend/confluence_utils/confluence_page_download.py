#%%
from atlassian import Confluence
from dotenv import load_dotenv, find_dotenv
import os

#%%
load_dotenv(find_dotenv())

confluence = Confluence(
    url=os.environ["atlassian_url"],
    username=os.environ["atlassian_username"],
    password=os.environ["atlassian_api_token"],
    api_version="cloud"
    # cloud=True
    )

#%%
# confluence.get_all_spaces(start=0, 
#                         limit=10, 
#                         expand=None)

# %%
id_list = [1130397701, 1376845825, 1287684100]
for id in id_list:
    page_info = confluence.get_page_by_id(page_id=id, expand=None, status=None, version=None)
    pdf_name = page_info["title"].replace(" ", "_")
    print(pdf_name)
    with open(f"Backend/confluence_utils/output_pdfs/{pdf_name}.pdf", "wb") as pdf_file:
        pdf_file.write(confluence.get_page_as_pdf(id))

# %%
