2. **Set Up a Virtual Environment**

   ### Windows:

   - Open a terminal in the repository directory.
   - Run the following commands:
     ```
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     ```

   ### Linux/Mac:

   - Open a terminal in the repository directory.
   - Run the following commands:
     ```
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     sudo apt-get install python3-tk
     ```

3. **Deactivate the Virtual Environment:**

   - To deactivate the virtual environment, simply run:
     ```
     deactivate
     ```

4. **Reactivating the Virtual Environment:**

   - Every time you return to the project, you need to activate the virtual environment before running any scripts:

     ### Windows:

     ```
     venv\Scripts\activate
     ```

     ### Linux/Mac:

     ```
     source venv/bin/activate
     ```

---
