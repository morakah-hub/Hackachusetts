// Start.exe: double-click launcher for Armstrong Studio (runs start.ps1 next to it and shows its progress).
// Build (any Windows 10/11, no SDK needed):
//   C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /nologo /target:exe /win32icon:launcher\logo.ico /out:Start.exe launcher\Start.cs
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

[assembly: AssemblyTitle("Armstrong Studio")]
[assembly: AssemblyProduct("Armstrong Studio")]
[assembly: AssemblyDescription("Starts the Armstrong Studio robot workcell simulation and opens it in the browser.")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.Title = "Armstrong Studio";
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(dir, "start.ps1");
        if (!File.Exists(script))
        {
            Console.WriteLine("start.ps1 was not found next to Start.exe.");
            Console.WriteLine("Keep Start.exe inside the Armstrong Studio folder (the one you downloaded or unzipped).");
            Pause();
            return 1;
        }
        string extra = string.Join(" ", args);
        var psi = new ProcessStartInfo("powershell.exe",
            "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\" " + extra)
        {
            UseShellExecute = false,
            WorkingDirectory = dir,
        };
        try
        {
            using (Process p = Process.Start(psi))
            {
                p.WaitForExit();
                if (p.ExitCode != 0)
                {
                    Pause();
                }
                return p.ExitCode;
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Could not start PowerShell: " + e.Message);
            Pause();
            return 1;
        }
    }

    private static void Pause()
    {
        Console.WriteLine();
        Console.WriteLine("Press any key to close this window.");
        try { Console.ReadKey(true); } catch (InvalidOperationException) { }
    }
}
