# Windows Save / Export failure: preserve the open session first

Keep the existing app and its PowerShell window open. Closing PowerShell can
terminate the app without saving. Do not restart the app, reopen the observation, or choose
"Open without saved labels" while the latest work exists only in that window.
Updating Python source files does not update the functions already loaded in a
running process. A restart to pick up the fix must come **after** recovery.

## Recover from the already-open app

Use the same Python environment that launches Mars Obs. The helper can be copied
into the old checkout's `scripts` directory; it works with its existing label
format and does not require restarting or reinstalling the running app.

1. Pause labeling. In a second terminal, from the project directory, run:

   ```powershell
   python scripts/rescue_open_session.py
   ```

   If Python is not activated, use the app environment's interpreter, for example
   `.\venv\Scripts\python.exe scripts\rescue_open_session.py`.

2. The helper creates a **new** folder under your Windows home directory's
   `MarsLabelRescue` folder. It configures delete protection on its `capture`
   subfolder, then tests file creation, writing, reading, flushing, and protection
   against deletion. It changes permissions only on this newly created folder.
   If setup fails, stop: keep the app open and retain the terminal error.

3. In the **existing open app**, select **File → Set Labels Folder…** and choose
   the exact `capture` folder printed by the helper. Dismiss the confirmation.
   **Do not follow the message's suggestion to reopen the observation.** Changing
   the destination alone retains the labels already in memory.

4. Click **Save labels** once and wait for the save attempt to finish. A save
   error is expected in this old process; it may now say **Permission denied**.
   The protected folder keeps the temporary Parquet that the failing save would
   otherwise delete. This process does not write into the original labels folder.

5. Return to the helper and press Enter. It copies the newest snapshot to a
   separate recovery folder, decodes every record, checks the complete grid,
   label statuses and block coordinates, and writes a checksum and count report.
   It prints **VERIFIED**, the labeled-block count, and the **Resume folder**.
   If verification fails, **keep the app open** and retain the error.

6. Compare the reported labeled-block count with the app and copy the verified
   recovery folder to a second safe location. Keep the original capture and any
   older `configs/labels` or custom labels folder. Do not merge or overwrite them.

7. Apply the code fix to the app installation (see below), then start a **second
   instance of the fixed app**, initially without an image.
   Choose **File → Set Labels Folder… → the printed Resume folder**, then open
   the exact original JP2/TIFF. Check the count and several known labels, including
   the most recent edits. Press **Save labels** and confirm the saved count/time,
   then test **Export Labels**. Only after checking this instance and the separate
   backup should you retire the old process. If its close operation still reports
   failure, close that old process via Task Manager only after the verified backup
   and fixed instance are confirmed; no further edits should have been made there.

The captured Parquet contains cursor, geometry, and source identity itself; a
missing `.session.json` sidecar does not prevent resuming. The capture folder is
only a rescue destination. Use the verified **Resume folder** for normal saves.
The helper prints a command to remove protection from the isolated capture folder
after recovery; don't delete it before you have independent verified copies.

## Apply the fix after capturing the labels

Update the checkout with the patched source files. The handoff ZIP contains the
three changed application files, this guide, and the rescue helper, **no label
data**. Extract it to a separate directory first. Its source files are based on
commit `2a2c4cc`; copy them into a matching checkout, or merge the changes if your
checkout has additional edits. Never replace or delete `configs/labels`, other
label folders, source images, or class definitions during this update.

The urgent storage fix by itself is one line in
`src/marslabeler/io/atomic.py`: replace `temporary.open("rb")` with
`temporary.open("r+b")`. Keep the `os.fsync` and `os.replace` calls. Do not use
`wb`, which truncates the file. The additional GUI changes are in `mainwindow.py`
and `exportdialog.py`. An editable installation (`pip install -e ...`) uses the
changed source when a **new** process starts. If the installation is not editable,
reinstall the fixed checkout into the same Python environment, for example:

```powershell
python -m pip install -e ".[gui]"
```

Keep the old process open while testing the second instance as described above.

To validate/copy an already retained Parquet or temporary file into another fresh
snapshot folder (also works on macOS/Linux):

```powershell
python scripts/rescue_open_session.py --recover-file "C:\path\to\snapshot.tmp" --output "C:\path\to\recovered"
```

## The fix

`atomic_output` used to reopen the completed temporary file as `rb` before
`os.fsync`. Windows requires a writable file handle for flushing; this can raise
`[Errno 9] Bad file descriptor`. The exception happens before the new Parquet
replaces the old file, so both saving and the export snapshot fail. Cleanup then
deletes the failed temporary file. Previous successful snapshots remain available,
but this does not prove that the current in-memory edits have ever been saved.

The fix opens the completed temporary file as `r+b` (without truncation), flushes
it, closes the handle, and only then atomically replaces the destination. The
file format and class mappings do not change. Save feedback now retains a count
and timestamp; failures display a dialog and retain a visible failure indicator.
Autosave failures notify once until a successful save, while manual retries show
their result again. Failed saves continue to block closing/switching observations.

References: Microsoft's [FlushFileBuffers handle requirements](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)
and [icacls delete/inheritance permissions](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/icacls).

## Verification scope

Local verification on 2026-10-09: **402 passed, 1 skipped** (the native Windows
ACL test). After the final save-feedback adjustment, all 52 affected GUI,
persistence, and recovery tests passed again (1 Windows-only skip).
The real HiRISE observation `ESP_092084_1525_RED.JP2` also completed save, reopen,
and GUI export with the writable-flush rule enforced: six 512×512 crops,
1,572,864 pixels with zero differences, correct label mappings, and a successful
small-model training step. Only isolated QA annotations were used. Artifacts are
in `exports/windows-save-verification-20261009/` (ignored by Git).

Regression tests impose Windows's writable-flush requirement on all platforms,
exercise the actual GUI Save and Export buttons, verify crop pixels and classes,
reopen saved sessions, and check that write/flush/replace failures preserve the
previous file and current in-memory edits. Rescue tests decode and reopen copied
snapshots and reproduce the old writer's failed cleanup under delete protection.

The Windows ACL test runs only on Windows. A GitHub Actions workflow now runs
these persistence/export/recovery tests on Windows with Python 3.11 and 3.12.
Adding that workflow is not evidence that it has run: local development/testing
for this fix is on macOS. The rescue helper therefore also checks the required
Windows permissions and Parquet writing on the affected PC before giving its
save instructions. The colleague's actual progress is not confirmed safe until
her snapshot verifies and reopens successfully.
