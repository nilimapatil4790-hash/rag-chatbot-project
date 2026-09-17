from pypdf import PdfReader

pdf_path = "document/sample.pdf"

reader = PdfReader(pdf_path)

print("PDF loaded successfully!")
print("Number of pages:", len(reader.pages))

text = reader.pages[0].extract_text()

print("\nFirst page content:\n")
print(text[:1000])