import { forwardRef } from "react";

const FileUploader = forwardRef(function FileUploader({ onFileChange }, ref) {
  return (
    <input
      type="file"
      accept="audio/*"
      ref={ref}
      onChange={onFileChange}
      className="hidden"
    />
  );
});

export default FileUploader;
